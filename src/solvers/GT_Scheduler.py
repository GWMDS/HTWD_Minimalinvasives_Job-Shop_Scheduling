import matplotlib.pyplot as plt
import random
from collections import defaultdict
from typing import Literal, List, Dict, Optional
from src.domain.Collection import LiveJobCollection
from src.domain.orm_models import JobOperation


class Scheduler:

    def __init__(self, jobs_collection: LiveJobCollection, schedule_start: int = 0):


        # JobsCollections and information
        self.jobs_collection = jobs_collection
        self.previous_schedule_jobs_collection = LiveJobCollection()
        self.active_jobs_collection = LiveJobCollection()

        self.machines = jobs_collection.get_unique_machine_names()
        self.machine_ready_time: Dict[str, int] = {m: schedule_start for m in self.machines}

        self.schedule_start = schedule_start

        self.total_ops = jobs_collection.count_operations()

        for job in self.jobs_collection.values():
            job.current_operation = job.get_first_operation()
            job.current_operation_earliest_start = max(job.earliest_start, schedule_start)


    def get_last_end_per_machine(self, collection: Optional[LiveJobCollection]) -> Dict[str, int]:
        """
        Liefert je Maschine die größte 'end'-Zeit aus der gegebenen LiveJobCollection.
        Operationen ohne gesetztes 'end' werden ignoriert.
        """
        if collection is None:
            return {}

        last_end: Dict[str, int] = {}
        for job in collection.values():
            for op in job.operations:
                if op.end is None:
                    continue
                m = op.machine_name
                e = int(op.end)
                if m not in last_end or e > last_end[m]:
                    last_end[m] = e
        return last_end

    def set_active_jobs_collection(self, active_jobs_collection: Optional[LiveJobCollection]):
        self.active_jobs_collection = active_jobs_collection

        last_end_per_machine = self.get_last_end_per_machine(self.active_jobs_collection)
        for m, e in last_end_per_machine.items():
            self.machine_ready_time[m] = max(self.machine_ready_time.get(m, self.schedule_start), e)

        for active_job in self.active_jobs_collection.values():
            job = self.jobs_collection.get(active_job.id)
            if job is None:
                continue

            last_op = active_job.get_last_operation()
            if last_op is not None:
                job.current_operation_earliest_start = max(self.schedule_start, last_op.end)


    def set_previous_schedule_jobs_collection(self, previous_schedule_jobs_collection: LiveJobCollection):
        self.previous_schedule_jobs_collection = previous_schedule_jobs_collection

    def select_by_priority(
            self,
            conflict_ops: List[JobOperation],
            rule: Literal["SPT", "FCFS", "MWKR", "EDD", "SLACK", "DEVIATION", "DEVIATION_INSERT"] = "SPT"
    ) -> Optional[JobOperation]:
        """
        Wählt aus JobOperation-Objekten gemäß Regel:
        - SPT: kürzeste Bearbeitungszeit
        - FCFS: kleinste Job-Ankunftszeit
        - EDD: früheste Job-Deadline
        """
        if not conflict_ops:
            return None

        def _duration(op: JobOperation):
            return op.duration

        def _job_earliest_start(op: JobOperation):
            return op.job_earliest_start

        def _job_arrival(op: JobOperation):
            return op.job_arrival if op.job_arrival is not None else 0

        def _job_due_date(op: JobOperation):
            return op.job_due_date if op.job_due_date is not None else 0

        def _job_total_dur(op: JobOperation):
            return op.job.sum_duration

        def _slack(op: JobOperation):
            return _job_due_date(op) - (op.start + _remaining_work(op))


        def _remaining_work(op: JobOperation):
            # inkl. aktueller Operation
            return op.job.sum_left_duration(op.position_number)

        def _start_deviation(op: JobOperation):
            prev_op_version = self.previous_schedule_jobs_collection.get_operation(op.job_id, op.position_number)
            if prev_op_version is not None:
                return prev_op_version.start - op.start
            else:
                return None

        def _random(_: object) -> float:
            # gibt einen Zufallswert zwischen 0 und 1 zurück
            return random.random()

        if rule == "SPT":
            key = lambda x: (_duration(x), _job_arrival(x), _job_earliest_start(x), _job_total_dur(x))
            return min(conflict_ops, key=key)

        elif rule == "FCFS":
            key = lambda x: (_job_arrival(x), _job_earliest_start(x), _duration(x), _job_total_dur(x))
            return min(conflict_ops, key=key)

        elif rule == "EDD":
            key = lambda x: (_job_due_date(x), _job_arrival(x), _job_earliest_start(x), _job_total_dur(x))
            return min(conflict_ops, key=key)

        elif rule == "MWKR":
            # Meiste Restarbeit zuerst, Zufall als Tie-Breaker
            key = lambda x: (_remaining_work(x), _random(x))
            return max(conflict_ops, key=key)

        elif rule == "MWKR_LPT":
            # Meiste Restarbeit zuerst
            key = lambda x: (_remaining_work(x), _duration(x))
            return max(conflict_ops, key=key)

        elif rule == "SLACK":
            # kleinste Slack zuerst; bei Gleichstand: frühester Start, dann SPT
            key = lambda x: (_slack(x), _job_earliest_start(x), _duration(x), x.job_id)
            return min(conflict_ops, key=key)

        elif rule == "DEVIATION":
            prev_ops = [op for op in conflict_ops if _start_deviation(op) is not None]

            # previous operations
            if not prev_ops:
                previous_operation = None
            else:
                key = lambda x: (_start_deviation(x), _slack(x), _job_earliest_start(x), _duration(x))
                previous_operation = min(prev_ops, key=key)

            # new operations
            new_ops = [op for op in conflict_ops if _start_deviation(op) is None]
            if not new_ops:
                new_operation = None
            else:
                key = lambda x: (_slack(x), _job_earliest_start(x), _duration(x))
                new_operation = min(new_ops, key=key)

            # select between new and previous
            if previous_operation is None:
                return new_operation
            else:
                return previous_operation


        elif rule == "DEVIATION_INSERT":
            prev_ops = [op for op in conflict_ops if _start_deviation(op) is not None]

            # previous operations
            if not prev_ops:
                previous_operation = None
            else:
                key = lambda x: (_start_deviation(x), _slack(x), _job_earliest_start(x), _duration(x))
                previous_operation = min(prev_ops, key=key)

            # new operations
            new_ops = [op for op in conflict_ops if _start_deviation(op) is None]
            if not new_ops:
                new_operation = None
            else:
                key = lambda x: (_slack(x), _job_earliest_start(x), _duration(x))
                new_operation = min(new_ops, key=key)

            # select between new and previous
            if new_operation is None:
                return previous_operation
            elif previous_operation is None:
                return new_operation
            else:
                next_start = new_operation.end
                prev_op_version = self.previous_schedule_jobs_collection.get_operation(
                    job_id=previous_operation.job_id,
                    position_number = previous_operation.position_number
                )
                next_deviation = prev_op_version.start - next_start
                if next_deviation <= 0:
                    return new_operation
                else:
                    return previous_operation


        else:
            raise ValueError("Invalid rule")


    def intervals_overlap(self, operation_a: JobOperation, operation_b: JobOperation) -> bool:
        # echte Überlappung (GT-Konfliktlogik)
        #if not (operation_a.end <= operation_b.start or operation_b.end <= operation_a.start):
        #    print(f"{operation_a.start = }, {operation_a.end = };  {operation_b.start = } {operation_b.end = }")

        return not (operation_a.end <= operation_b.start or operation_b.end <= operation_a.start)


    def get_machine_candidates(self) -> Dict[str, list[JobOperation]]:
        machine_candidates: Dict[str, list[JobOperation]] = defaultdict(list)
        for job in self.jobs_collection.values():
            operation = job.current_operation
            if operation is not None:
                operation.start = max(self.machine_ready_time[operation.machine_name], job.current_operation_earliest_start)
                operation.end = operation.start + operation.duration
                machine_candidates[operation.machine_name].append(operation)
        return dict(machine_candidates)


    def get_schedule(
        self,
        priority_rule: Literal[
            "SPT", "FCFS", "EDD", "MWKR", "SLACK", "DEVIATION", "DEVIATION_INSERT"
        ] = "SPT",
        add_overlap_to_conflict: bool = False,
        plot_gantt: bool = False,
        gantt_path: Optional[str] = None,
    ):
 

        schedule_job_collection = LiveJobCollection()
        planned = 0

        def _has_unscheduled_ops() -> bool:
            """Prüft, ob noch Operationen ohne Startzeit existieren."""
            for job in self.jobs_collection.values():
                for op in job.operations:
                    if op.start is None:
                        return True
            return False

        # ------------------------------------------------------
        # Hauptschleife
        # ------------------------------------------------------
        while planned < self.total_ops and _has_unscheduled_ops():

            # 1) nächste planbare Operationen pro Job finden
            next_ops = []  # (job, idx, earliest_start, op)

            for job in self.jobs_collection.values():
                ops = job.operations
                for idx, op in enumerate(ops):
                    if op.start is None:
                        # Vorgänger-Ende
                        if idx > 0:
                            prev_op = ops[idx - 1]
                            if prev_op.end is None:
                                break  # Vorgänger nicht geplant → Job blockiert
                            earliest_start = int(prev_op.end)
                        else:
                            earliest_start = max(
                                job.current_operation_earliest_start,
                                job.earliest_start,
                                self.schedule_start,
                            )

                        next_ops.append((job, idx, earliest_start, op))
                        break  # pro Job nur die erste ungeschedulte Operation

            if not next_ops:
                break

            # 2) Konfliktmengen pro Maschine
            conflict_ops_per_machine: Dict[str, list] = {}

            for job, idx, earliest_start, op in next_ops:
                m = op.machine_name
                m_available = self.machine_ready_time.get(m, self.schedule_start)

                start_time = max(earliest_start, m_available)
                end_time = start_time + op.duration

                if m not in conflict_ops_per_machine:
                    conflict_ops_per_machine[m] = []
                conflict_ops_per_machine[m].append(
                    (job, idx, start_time, end_time, op)
                )

            # 3) global dmin = früheste Endzeit aller Kandidaten
            all_candidates = [
                (job, idx, start_time, end_time, op)
                for lst in conflict_ops_per_machine.values()
                for (job, idx, start_time, end_time, op) in lst
            ]
            if not all_candidates:
                break

            dmin = min(end_time for (_, _, _, end_time, _) in all_candidates)

            # 4) pro Maschine: Konfliktmenge K_m und Prioritätswahl
            selected_per_machine = []  # (op, end_time)

            for m, candidates in conflict_ops_per_machine.items():
                # Konfliktmenge K_m = alle Ops mit Start < dmin
                K = [
                    (job, idx, start_time, end_time, op)
                    for (job, idx, start_time, end_time, op) in candidates
                    if start_time < dmin
                ]
                if not K:
                    continue

                # start/end in Operation speichern
                conflict_ops = []
                for job, idx, start_time, end_time, op in K:
                    op.start = start_time
                    op.end = end_time
                    conflict_ops.append(op)

                # Auswahl nach Prioritätsregel
                selected_op = self.select_by_priority(conflict_ops, priority_rule)
                if selected_op is None:
                    continue

                # Endzeit des ausgewählten Ops besorgen
                selected_end = None
                for _, _, s_start, s_end, s_op in K:
                    if s_op is selected_op:
                        selected_end = s_end
                        break

                if selected_end is None:
                    continue

                selected_per_machine.append((selected_op, selected_end))

            if not selected_per_machine:
                break

            # 5) global die Operation mit kleinster Endzeit einplanen
            op_to_schedule, op_end = min(selected_per_machine, key=lambda t: t[1])
            job = op_to_schedule.job
            machine = op_to_schedule.machine_name

            # Falls start/end nicht gesetzt (shouldn't happen)
            if op_to_schedule.start is None or op_to_schedule.end is None:
                m_available = self.machine_ready_time.get(machine, self.schedule_start)
                earliest_start = max(
                    job.current_operation_earliest_start,
                    job.earliest_start,
                    self.schedule_start,
                )
                start_time = max(earliest_start, m_available)
                op_to_schedule.start = start_time
                op_to_schedule.end = start_time + op_to_schedule.duration
                op_end = op_to_schedule.end

            # fest einplanen
            schedule_job_collection.add_operation_instance(op_to_schedule)
            planned += 1

            # Maschine blockieren
            self.machine_ready_time[machine] = int(op_end)

            # nächste Operation des Jobs vorbereiten
            next_op = job.get_next_operation(op_to_schedule.position_number)
            job.current_operation = next_op
            if next_op is not None:
                job.current_operation_earliest_start = int(op_end)

        # ------------------------------------------------------
        # Optional: Gantt-Diagramm erzeugen
        # ------------------------------------------------------
        if plot_gantt:
            # Schedule in flache Liste bringen: (job_id, op_pos, machine, start, end)
            gantt_rows = []
            for job in schedule_job_collection.values():
                for op in job.operations:
                    if op.start is None or op.end is None:
                        continue
                    gantt_rows.append(
                        (
                            job.id,
                            op.position_number,
                            op.machine_name,
                            int(op.start),
                            int(op.end),
                        )
                    )

            if gantt_rows:
                # nach Maschine, dann Start sortieren
                gantt_rows.sort(key=lambda x: (x[2], x[3]))

                machines = sorted({row[2] for row in gantt_rows})
                job_ids = sorted({row[0] for row in gantt_rows})

                colors_palette = [
                    "tab:blue",
                    "tab:orange",
                    "tab:green",
                    "tab:red",
                    "tab:purple",
                    "tab:brown",
                    "tab:pink",
                    "tab:gray",
                    "tab:olive",
                    "tab:cyan",
                ]
                job_colors = {
                    job_id: colors_palette[i % len(colors_palette)]
                    for i, job_id in enumerate(job_ids)
                }

                fig, ax = plt.subplots(figsize=(12, 6))

                for job_id, op_pos, m, start, end in gantt_rows:
                    color = job_colors[job_id]
                    ax.barh(
                        f"Maschine {m}",
                        end - start,
                        left=start,
                        color=color,
                        edgecolor="black",
                    )
                    ax.text(
                        start + (end - start) / 2,
                        f"Maschine {m}",
                        f"Job {job_id}",
                        va="center",
                        ha="center",
                        color="white",
                        fontsize=9,
                    )

                ax.set_xlabel("Zeit")
                ax.set_ylabel("Maschinen")
                ax.set_title(f"Gantt-Diagramm – GT ({priority_rule})")
                ax.grid(True, axis="x", linestyle="--", alpha=0.6)
                plt.tight_layout()

                if gantt_path:
                    plt.savefig(gantt_path, dpi=300)
                    print(f"Gantt-Diagramm gespeichert als {gantt_path}")

                plt.show()

        return schedule_job_collection

