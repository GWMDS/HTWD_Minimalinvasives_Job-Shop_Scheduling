import random
import bisect
from collections import defaultdict
from typing import Literal, List, Dict, Optional
from src.domain.Collection import LiveJobCollection
from src.domain.orm_models import JobOperation


class Scheduler:

    def __init__(self, jobs_collection: LiveJobCollection, schedule_start: int = 0, schedule_end: Optional[int] = None):
        self.jobs_collection = jobs_collection
        self.previous_schedule_jobs_collection = LiveJobCollection()
        self.active_jobs_collection = LiveJobCollection()

        # Cache für MACHINE_SLOT Regel
        self.machine_slots: Dict[str, List[int]] = defaultdict(list)

        # Maschinennamen bereinigen
        raw_machines = jobs_collection.get_unique_machine_names()
        self.machines = [str(m).strip() for m in raw_machines]
        
        self.machine_ready_time: Dict[str, int] = {m: schedule_start for m in self.machines}
        
        self.schedule_start = schedule_start
        # NEU: Das Ende der Schicht (Hard Limit). Wenn None -> Unendlich.
        self.schedule_end = schedule_end if schedule_end is not None else float('inf')
        
        self.total_ops = jobs_collection.count_operations()

        # Reset aller Operationen (Dirty State Prevention)
        for job in self.jobs_collection.values():
            for op in job.operations:
                if op.machine_name:
                    op.machine_name = str(op.machine_name).strip()
                op.start = None
                op.end = None
                # Temporäre Attribute aufräumen
                if hasattr(op, 'temp_start'): del op.temp_start
                if hasattr(op, 'temp_end'): del op.temp_end
                if hasattr(op, '_tmp_prev_start'): del op._tmp_prev_start

            if job.operations:
                job.current_operation = job.operations[0]
                job.current_operation_earliest_start = max(job.earliest_start, schedule_start)

    def get_last_end_per_machine(self, collection: Optional[LiveJobCollection]) -> Dict[str, int]:
        if collection is None: return {}
        last_end: Dict[str, int] = {}
        for job in collection.values():
            for op in job.operations:
                if op.end is None: continue
                m = str(op.machine_name).strip()
                e = int(op.end)
                if m not in last_end or e > last_end[m]:
                    last_end[m] = e
        return last_end

    def set_active_jobs_collection(self, active_jobs_collection: Optional[LiveJobCollection]):
        self.active_jobs_collection = active_jobs_collection
        last_end_per_machine = self.get_last_end_per_machine(self.active_jobs_collection)
        for m, e in last_end_per_machine.items():
            clean_m = str(m).strip()
            current_ready = self.machine_ready_time.get(clean_m, self.schedule_start)
            self.machine_ready_time[clean_m] = max(current_ready, e)

        if self.active_jobs_collection:
            for active_job in self.active_jobs_collection.values():
                job = self.jobs_collection.get(active_job.id)
                if job is None: continue
                last_op = active_job.get_last_operation()
                if last_op and last_op.end:
                    job.current_operation_earliest_start = max(self.schedule_start, int(last_op.end))

    def set_previous_schedule_jobs_collection(self, previous_schedule_jobs_collection: LiveJobCollection):
        self.previous_schedule_jobs_collection = previous_schedule_jobs_collection
        
        # Slots für MACHINE_SLOT vorberechnen
        self.machine_slots = defaultdict(list)
        for job in previous_schedule_jobs_collection.values():
            for op in job.operations:
                if op.start is not None and op.machine_name:
                    m_name = str(op.machine_name).strip()
                    self.machine_slots[m_name].append(int(op.start))
        
        for m in self.machine_slots:
            self.machine_slots[m].sort()

    def select_by_priority(self, conflict_ops: List[JobOperation], rule: str) -> Optional[JobOperation]:
        if not conflict_ops: return None
        if len(conflict_ops) == 1: return conflict_ops[0]
        rule = rule.upper()

        def _duration(op): return op.duration
        def _arrival(op): return op.job_arrival if op.job_arrival else 0
        def _due_date(op): return op.job_due_date if op.job_due_date else 0
        def _job_earliest_start(op): return op.job_earliest_start
        def _rem_work(op): return op.job.sum_left_duration(op.position_number)
        def _job_id(op): return op.job_id
        
        def _slack(op): 
            if op.start is None: return 999999
            return _due_date(op) - (op.start + _rem_work(op))
        
        def _start_deviation(op):
            prev_op = self.previous_schedule_jobs_collection.get_operation(op.job_id, op.position_number)
            if prev_op and op.start is not None:
                return prev_op.start - op.start
            return None
            
        def _distance_to_slot(op):
            if op.start is None: return 999999
            m_name = str(op.machine_name).strip()
            slots = self.machine_slots.get(m_name, [])
            if not slots: return 0
            idx = bisect.bisect_left(slots, op.start)
            diff = float('inf')
            if idx > 0: diff = min(diff, abs(op.start - slots[idx - 1]))
            if idx < len(slots): diff = min(diff, abs(op.start - slots[idx]))
            return diff

        # --- REGELN ---
        if rule == "DEVIATION_INSERT":
            k_old, k_new = [], []
            for op in conflict_ops:
                prev_op = self.previous_schedule_jobs_collection.get_operation(op.job_id, op.position_number)
                if prev_op and prev_op.start is not None:
                    op._tmp_prev_start = prev_op.start
                    k_old.append(op)
                else: k_new.append(op)
            best_old = min(k_old, key=lambda x: (x._tmp_prev_start, _job_id(x))) if k_old else None
            best_new = min(k_new, key=lambda x: (_duration(x), _job_id(x))) if k_new else None
            if best_old and best_new:
                if best_new.end <= best_old._tmp_prev_start: return best_new
                return best_old
            return best_old or best_new

        elif rule == "DEVIATION":
            k_old, k_new = [], []
            for op in conflict_ops:
                prev_op = self.previous_schedule_jobs_collection.get_operation(op.job_id, op.position_number)
                if prev_op and prev_op.start is not None:
                    op._tmp_prev_start = prev_op.start
                    k_old.append(op)
                else: k_new.append(op)
            if k_old: return min(k_old, key=lambda x: (x._tmp_prev_start, _job_id(x)))
            if k_new: return min(k_new, key=lambda x: (_slack(x), _duration(x), _job_id(x)))

        elif rule == "WRC":
            ALPHA = 0.8
            def _wrc_score(op):
                if _due_date(op): latest_start = _due_date(op) - _rem_work(op)
                else: latest_start = 999999
                prev_op = self.previous_schedule_jobs_collection.get_operation(op.job_id, op.position_number)
                ref_start = prev_op.start if (prev_op and prev_op.start is not None) else latest_start
                return ALPHA * ref_start + (1 - ALPHA) * latest_start
            return min(conflict_ops, key=lambda x: (_wrc_score(x), _job_earliest_start(x)))

        elif rule == "MACHINE_SLOT":
            return min(conflict_ops, key=lambda x: (_distance_to_slot(x), _slack(x)))

        elif rule == "SPT": return min(conflict_ops, key=lambda x: (_duration(x), _arrival(x), _job_id(x)))
        elif rule == "FCFS": return min(conflict_ops, key=lambda x: (_arrival(x), _duration(x), _job_id(x)))
        elif rule == "EDD": return min(conflict_ops, key=lambda x: (_due_date(x), _arrival(x), _job_id(x)))
        elif rule == "MWKR": return max(conflict_ops, key=lambda x: (_rem_work(x), -_duration(x), _job_id(x)))
        elif rule == "SLACK": return min(conflict_ops, key=lambda x: (_slack(x), _duration(x), _job_id(x)))

        return conflict_ops[0]

    def get_schedule(self, priority_rule: str = "SPT", add_overlap_to_conflict: bool = True):
        schedule_result = LiveJobCollection()
        planned = 0
        loop_guard = 0
        max_loops = self.total_ops * 5

        while True:
            loop_guard += 1
            if loop_guard > max_loops: break

            machine_candidates = defaultdict(list)
            all_valid_candidates = []
            
            for job in self.jobs_collection.values():
                op = job.current_operation
                if op is not None:
                    est = max(self.machine_ready_time[op.machine_name], job.current_operation_earliest_start)
                    eft = est + op.duration
                    
                    # --- HARD CUT ---
                    # Wenn die Operation nicht mehr in die Schicht passt, ignorieren wir sie für diese Runde.
                    # Sie wird dann vom Experiment-Skript in den Backlog für die nächste Schicht gepackt.
                    if eft > self.schedule_end:
                        continue 

                    op.start = est
                    op.end = eft
                    machine_candidates[op.machine_name].append(op)
                    all_valid_candidates.append(op)

            if not all_valid_candidates: break

            min_eft = min(op.end for op in all_valid_candidates)
            target_machine = None
            for m, ops in machine_candidates.items():
                if any(o.end == min_eft for o in ops):
                    target_machine = m
                    break
            
            if target_machine is None: break

            ops_on_m = machine_candidates[target_machine]
            conflict_ops = [o for o in ops_on_m if o.start < min_eft]
            if not conflict_ops: conflict_ops = [o for o in ops_on_m if o.end == min_eft]

            selected_op = self.select_by_priority(conflict_ops, priority_rule)

            if selected_op:
                job = selected_op.job
                job.current_operation = job.get_next_operation(selected_op.position_number)
                job.current_operation_earliest_start = selected_op.end
                
                self.machine_ready_time[selected_op.machine_name] = selected_op.end
                schedule_result.add_operation_instance(selected_op)
                
                # Cleanup nicht gewählte Kandidaten
                for op in all_valid_candidates:
                    if op != selected_op:
                        op.start = None; op.end = None
            else:
                break
            
        return schedule_result