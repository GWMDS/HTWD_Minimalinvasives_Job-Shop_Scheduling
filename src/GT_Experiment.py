from decimal import Decimal
from typing import Literal
from pathlib import Path
import copy

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 

# Import für das Plotting (Fallback-sicher)
try:
    from src.analyses.fig_gantt import DataFramePlotGenerator
except ImportError:
    from src.DataFrameAnalyses import DataFramePlotGenerator

from config.project_config import get_solver_logs_path

from src.domain.Collection import LiveJobCollection
from src.domain.orm_models import LiveJob, JobOperation
from src.domain.Initializer import ExperimentInitializer
from src.domain.Query import JobQuery, MachineInstanceQuery, ExperimentQuery
from src.simulation.LognormalFactorGenerator import LognormalFactorGenerator
from src.simulation.ProductionSimulation import ProductionSimulation
from src.solvers.GT_Scheduler import Scheduler


def schedule_to_dataframe(schedule_jobs_collection: LiveJobCollection) -> pd.DataFrame:
    """Wandelt den aktuellen Schedule in einen DataFrame um."""
    rows = []
    for job in schedule_jobs_collection.values():
        
        # --- ÄNDERUNG: Wir nehmen die echte Job-ID statt der Routing-ID ---
        # ALT: routing_id = getattr(job, "routing_id", job.id)
        job_label = job.id 
        # ------------------------------------------------------------------
        
        for op in job.operations:
            if op.start is None or op.end is None:
                continue

            rows.append({
                "Job": job_label,  # Hier stand vorher routing_id
                "Machine": op.machine_name,
                "Operation": op.position_number,
                "Start": int(op.start),
                "Processing Time": int(op.duration),
                # "End": int(op.end) 
            })
    return pd.DataFrame(rows)


def create_schedule_snapshot(collection: LiveJobCollection) -> LiveJobCollection:
    """
    Erstellt eine tiefe Kopie des Schedules für die nächste Schicht (Previous Plan).
    Dies verhindert, dass der Scheduler beim Reset die historischen Daten löscht.
    """
    snapshot = LiveJobCollection()
    for job in collection.values():
        new_job = LiveJob(
            id=job.id,
            routing_id=job.routing_id,
            arrival=job.arrival,
            due_date=job.due_date
        )
        for op in job.operations:
            # Nur geplante Ops kopieren
            if op.start is not None:
                new_op = JobOperation(
                    job=new_job,
                    position_number=op.position_number,
                    machine_name=op.machine_name,
                    duration=op.duration,
                    start=op.start,
                    end=op.end
                )
                new_job.operations.append(new_op)
        
        # FIX: Dictionary-Syntax statt .add_job()
        snapshot[new_job.id] = new_job
        
    return snapshot


def run_experiment(
    experiment_id: int,
    shift_length: int,
    total_shift_number: int,
    priority_rule: Literal["SLACK", "DEVIATION"],
    source_name: str,
    max_bottleneck_utilization: Decimal,
    sim_sigma: float,
) -> None:

    # 1. Simulation Setup
    simulation = ProductionSimulation(verbose=False, shift_length=shift_length)

    # 2. PDF Setup
    logs_root = get_solver_logs_path()
    if isinstance(logs_root, str): out_dir = Path(logs_root)
    else: out_dir = logs_root
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = out_dir / f"gantt_experiment_{experiment_id}_{priority_rule}.pdf"
    pdf = PdfPages(pdf_path)
    print(f"--> Gantt-Chart wird erstellt in: {pdf_path}")

    # 3. Daten Laden & LiveJob Konvertierung
    orm_jobs = JobQuery.get_by_source_name_max_util_and_lt_arrival(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
        arrival_limit=60 * 24 * total_shift_number,
    )
    
    # ORM -> LiveJob (statische Listen erzeugen)
    live_jobs_list = []
    for j in orm_jobs:
        lj = LiveJob(
            id=j.id, routing_id=j.routing_id, arrival=j.arrival, 
            due_date=j.due_date, max_bottleneck_utilization=j.max_bottleneck_utilization
        )
        real_ops = []
        for op in j.operations:
            new_op = JobOperation(
                job=lj, position_number=op.position_number, machine_name=op.machine_name,
                duration=op.duration, transition_time=op.transition_time
            )
            real_ops.append(new_op)
        lj.operations = real_ops
        live_jobs_list.append(lj)

    jobs_collection = LiveJobCollection(live_jobs_list)

    # Maschinen & Rüstzeiten
    machines_instances = MachineInstanceQuery.get_by_source_name_and_max_bottleneck_utilization(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
    )
    for machine_instance in machines_instances:
        inst_name = str(machine_instance.name).strip()
        for job in jobs_collection.values():
            for operation in job.operations:
                if str(operation.machine_name).strip() == inst_name:
                    operation.transition_time = machine_instance.transition_time

    # Simulation Durations
    factor_gen = LognormalFactorGenerator(sigma=sim_sigma, seed=42)
    jobs_collection.sort_jobs_by_id()
    jobs_collection.sort_operations()
    for job in jobs_collection.values():
        for operation in job.operations:
            op_dur = operation.duration * factor_gen.sample()
            operation.sim_duration = int(op_dur)

    # Collections
    previous_schedule_snapshot = LiveJobCollection()
    active_job_ops_collection = LiveJobCollection()
    waiting_job_ops_collection = LiveJobCollection()

    # --- Shift Loop ---
    for shift_number in range(1, total_shift_number + 1):
        shift_start = (shift_number - 1) * shift_length
        shift_end = shift_number * shift_length
        print(f"Experiment {experiment_id} shift {shift_number}: {shift_start} to {shift_end}")

        # A) Job-Auswahl
        new_jobs = jobs_collection.get_subset_by_earliest_start(earliest_start=shift_start)
        
        combined_jobs = []
        combined_jobs.extend(new_jobs.values())
        
        # Slicing für wartende Jobs (nur noch nicht erledigte Ops)
        sim_finished = simulation.get_entire_finished_operation_collection()
        sim_active = simulation.get_active_operation_collection()
        
        for wait_job in waiting_job_ops_collection.values():
            max_done_pos = -1
            if wait_job.id in sim_finished:
                for op in sim_finished[wait_job.id].operations:
                    if op.position_number > max_done_pos: max_done_pos = op.position_number
            if wait_job.id in sim_active:
                for op in sim_active[wait_job.id].operations:
                    if op.position_number > max_done_pos: max_done_pos = op.position_number
            
            future_ops = [op for op in wait_job.operations if op.position_number > max_done_pos]
            
            if future_ops:
                sliced_job = copy.copy(wait_job) 
                sliced_job.operations = future_ops 
                sliced_job.current_operation = None 
                combined_jobs.append(sliced_job)

        current_jobs_collection = LiveJobCollection(combined_jobs)

        # B) Scheduling
        scheduler = Scheduler(
            jobs_collection=current_jobs_collection,
            schedule_start=shift_start,
        )
        scheduler.set_active_jobs_collection(active_job_ops_collection)
        scheduler.set_previous_schedule_jobs_collection(previous_schedule_snapshot)

        # Planen
        current_schedule_result = scheduler.get_schedule(priority_rule=priority_rule)

        # Snapshot für nächste Runde sichern
        previous_schedule_snapshot = create_schedule_snapshot(current_schedule_result)

        # C) Speichern & Plotten
        ExperimentQuery.save_schedule_jobs(
            experiment_id=experiment_id,
            shift_number=shift_number,
            live_jobs=current_schedule_result.values(),
        )

        # --- AUTOMATISCHES GANTT DIAGRAMM ---
        df_shift = schedule_to_dataframe(current_schedule_result)
        
        if not df_shift.empty:
            df_plot = df_shift.copy()
            # Relative Startzeit
            if "Start" in df_plot.columns:
                df_plot["Start"] = df_plot["Start"] - shift_start
            
            try:
                fig = DataFramePlotGenerator.get_gantt_chart_figure(
                    df_workflow=df_plot,
                    title=f"Exp {experiment_id} ({priority_rule}) - Shift {shift_number} (t={shift_start}-{shift_end})",
                    job_column="Job",
                    machine_column="Machine",
                    duration_column="Processing Time",
                    perspective="Machine"
                )
                pdf.savefig(fig)
                plt.close(fig) 
            except Exception as e:
                print(f"Fehler beim Plotten Shift {shift_number}: {e}")
        else:
            print(f"-> Keine Operationen in Shift {shift_number} geplant.")
        # ------------------------------------

        # D) Simulation
        if shift_number == 1:
            simulation.initialize_run(schedule_collection=current_schedule_result, start_time=shift_start)
        else:
            simulation.continue_run(schedule_collection=current_schedule_result)

        active_job_ops_collection = simulation.get_active_operation_collection()
        waiting_job_ops_collection = simulation.get_waiting_operation_collection()

    # Ende
    entire_sim = simulation.get_entire_finished_operation_collection()
    ExperimentQuery.save_simulation_jobs(experiment_id=experiment_id, live_jobs=entire_sim.values())
    
    pdf.close()
    print(f"Fertig! Alle Diagramme gespeichert in: {pdf_path}")


def init_experiment(
    shift_length: int,
    total_shift_number: int,
    priority_rule: Literal["SLACK", "DEVIATION"],
    source_name: str,
    max_bottleneck_utilization: Decimal,
    sim_sigma: float,
) -> int:

    experiment_id = ExperimentInitializer.insert_experiment(
        source_name=source_name,
        absolute_lateness_ratio=0,
        inner_tardiness_ratio=0,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization:.2f}"),
        sim_sigma=sim_sigma,
        experiment_type=f"GT_{priority_rule}",
    )

    run_experiment(
        experiment_id=experiment_id,
        shift_length=shift_length,
        total_shift_number=total_shift_number,
        priority_rule=priority_rule,
        source_name=source_name,
        max_bottleneck_utilization=max_bottleneck_utilization,
        sim_sigma=sim_sigma,
    )
    
    return experiment_id