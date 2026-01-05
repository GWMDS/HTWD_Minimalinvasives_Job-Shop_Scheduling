import random
from collections import defaultdict
from typing import Literal, List, Dict, Optional
from src.domain.Collection import LiveJobCollection
from src.domain.orm_models import JobOperation


class Scheduler:

    def __init__(self, jobs_collection: LiveJobCollection, schedule_start: int = 0):
        """
        Initialisiert den Scheduler für ein bestimmtes Zeitfenster.
        """
        self.jobs_collection = jobs_collection
        self.previous_schedule_jobs_collection = LiveJobCollection()
        self.active_jobs_collection = LiveJobCollection()

        self.machines = jobs_collection.get_unique_machine_names()
        # Maschinen sind erst ab Schichtbeginn (schedule_start) bereit
        self.machine_ready_time: Dict[str, int] = {m: schedule_start for m in self.machines}

        self.schedule_start = schedule_start
        self.total_ops = jobs_collection.count_operations()

        # Jobs initialisieren
        for job in self.jobs_collection.values():
            if job.current_operation is None:
                job.current_operation = job.get_first_operation()
            
            job_arrival = job.earliest_start if job.earliest_start is not None else 0
            job.current_operation_earliest_start = max(job_arrival, schedule_start)

    def get_last_end_per_machine(self, collection: Optional[LiveJobCollection]) -> Dict[str, int]:
        if collection is None:
            return {}
        last_end: Dict[str, int] = {}
        for job in collection.values():
            for op in job.operations:
                if op.end is None: continue
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
            if job is None: continue
            last_op = active_job.get_last_operation()
            if last_op is not None:
                job.current_operation_earliest_start = max(self.schedule_start, last_op.end)

    def set_previous_schedule_jobs_collection(self, previous_schedule_jobs_collection: LiveJobCollection):
        self.previous_schedule_jobs_collection = previous_schedule_jobs_collection

    def select_by_priority(
            self,
            conflict_ops: List[JobOperation],
            rule: str = "SPT"
    ) -> Optional[JobOperation]:
        """
        Entscheidungslogik basierend auf Prioritätsregeln.
        """
        if not conflict_ops:
            return None
        
        if len(conflict_ops) == 1:
            return conflict_ops[0]

        # Normalisierung der Regel
        rule = rule.upper()

        # --- Hilfsmetriken ---
        def _duration(op: JobOperation): return op.duration
        def _arrival(op: JobOperation): return op.job_arrival if op.job_arrival is not None else 0
        def _due_date(op: JobOperation): return op.job_due_date if op.job_due_date is not None else 0
        def _rem_work(op: JobOperation): return op.job.sum_left_duration(op.position_number)
        def _job_id(op: JobOperation): return op.job_id

        # --- Regeln ---

        if rule == "DEVIATION_INSERT":
            k_old, k_new = [], []
            for op in conflict_ops:
                prev_op = self.previous_schedule_jobs_collection.get_operation(op.job_id, op.position_number)
                if prev_op and prev_op.start is not None:
                    op._tmp_prev_start = prev_op.start
                    k_old.append(op)
                else:
                    k_new.append(op)
            
            best_old = min(k_old, key=lambda x: (x._tmp_prev_start, _job_id(x))) if k_old else None
            best_new = min(k_new, key=lambda x: (_duration(x), _job_id(x))) if k_new else None
            
            if best_old and best_new:
                if best_new.end <= best_old._tmp_prev_start:
                    return best_new
                return best_old
            return best_old or best_new

        elif rule == "SPT":
            # Shortest Processing Time
            return min(conflict_ops, key=lambda x: (_duration(x), _arrival(x), _job_id(x)))

        elif rule == "FCFS":
            # First Come First Served (Job Arrival)
            return min(conflict_ops, key=lambda x: (_arrival(x), _duration(x), _job_id(x)))

        elif rule == "EDD":
            # Earliest Due Date
            return min(conflict_ops, key=lambda x: (_due_date(x), _arrival(x), _job_id(x)))

        elif rule == "MWKR":
            # Most Work Remaining
            return max(conflict_ops, key=lambda x: (_rem_work(x), -_duration(x), _job_id(x)))

        elif rule == "SLACK":
            # Slack = (Due Date) - (Current Start + Remaining Work)
            # Je kleiner der Slack, desto dringender
            def calc_slack(op): return _due_date(op) - (op.start + _rem_work(op))
            return min(conflict_ops, key=lambda x: (calc_slack(x), _duration(x), _job_id(x)))

        # Fallback auf erstes Element
        return conflict_ops[0]

    def get_schedule(self, priority_rule: str = "SPT", shift_end: Optional[int] = None):
        schedule_result = LiveJobCollection()
        
        # Metriken für das Debugging
        total_decisions = 0
        real_choices = 0
        
        while True:
            machine_candidates = defaultdict(list)
            all_valid_candidates = []
            
            for job in self.jobs_collection.values():
                op = job.current_operation
                if op is not None:
                    est = max(self.machine_ready_time[op.machine_name], job.current_operation_earliest_start)
                    eft = est + op.duration
                    
                    if shift_end is None or eft <= shift_end:
                        op.start = est
                        op.end = eft
                        machine_candidates[op.machine_name].append(op)
                        all_valid_candidates.append(op)

            if not all_valid_candidates:
                break

            # Giffler-Thompson: Finde globales EFT
            min_eft = min(op.end for op in all_valid_candidates)
            
            # Maschine identifizieren
            target_machine = None
            for m, ops in machine_candidates.items():
                if any(o.end == min_eft for o in ops):
                    target_machine = m
                    break
            
            if target_machine is None: break

            # Konfliktmenge auf dieser Maschine
            ops_on_m = machine_candidates[target_machine]
            conflict_ops = [o for o in ops_on_m if o.start < min_eft]
            
            if not conflict_ops:
                conflict_ops = [o for o in ops_on_m if o.end == min_eft]

            # Wahl-Statistik
            total_decisions += 1
            if len(conflict_ops) > 1:
                real_choices += 1

            selected_op = self.select_by_priority(conflict_ops, priority_rule)

            if selected_op:
                job = selected_op.job
                job.current_operation = job.get_next_operation(selected_op.position_number)
                job.current_operation_earliest_start = selected_op.end

                self.machine_ready_time[selected_op.machine_name] = selected_op.end
                schedule_result.add_operation_instance(selected_op)
            else:
                break
            
        # --- DEBUG OUTPUT ---
        print(f"\n--- SCHEDULER RESULT CHECK (SHIFT: {self.schedule_start} - {shift_end}) ---")
        print(f"Regel: {priority_rule} | Entscheidungen: {total_decisions} | Echte Wahlmöglichkeiten: {real_choices}")
        
        all_scheduled = []
        for job in schedule_result.values():
            for op in job.operations:
                if op.start is not None:
                    all_scheduled.append(op)
        
        all_scheduled.sort(key=lambda x: x.start)
        for op in all_scheduled:
            print(f"[{op.start} - {op.end}] Job {op.job_id} Op {op.position_number} on {op.machine_name}")
        
        if not all_scheduled:
            print("KEINE Operationen geplant.")
            
        return schedule_result