from src.domain.Collection import LiveJobCollection
from collections import defaultdict
import bisect

def calculate_stability_metrics(
        current_schedule: LiveJobCollection, 
        previous_schedule: LiveJobCollection, 
        ignore_threshold: int = 120
    ):
    """
    Berechnet Metriken zur Maschinen-Stabilität.
    
    Filter-Logik:
    Es werden nur Operationen im aktuellen Plan bewertet, die sich in der Nähe 
    (innerhalb 'ignore_threshold', default 120 Min) eines alten Slots befinden. 
    Komplett neue Slots (die weit weg von alten Mustern liegen) werden ignoriert.

    Metriken:
    1. SlotDev_Avg: Durchschnittl. Abstand der relevanten Ops zu ihrem alten Slot.
    2. SlotHit_Rate: % der relevanten Ops, die den Slot sehr gut treffen (<= 50 Min).
    """
    
    total_slot_distance = 0
    slot_hits = 0
    total_relevant_ops = 0  # Zählt nur Ops, die nahe an einem alten Slot liegen
    
    # 1. Datenvorbereitung: Alte Slots sammeln
    prev_slots_map = defaultdict(list)
    for job in previous_schedule.values():
        for op in job.operations:
            if op.start is not None:
                m_name = str(op.machine_name).strip()
                prev_slots_map[m_name].append(int(op.start))
    
    for m in prev_slots_map:
        prev_slots_map[m].sort()

    # 2. Iteration über den NEUEN Plan
    for job in current_schedule.values():
        for op in job.operations:
            if op.start is None: continue
            
            m_name = str(op.machine_name).strip()
            old_slots = prev_slots_map.get(m_name, [])
            
            if not old_slots:
                continue 

            s_new = int(op.start)
            
            # Finde nächstgelegenen alten Slot
            idx = bisect.bisect_left(old_slots, s_new)
            
            diff = float('inf')
            
            # Nachbar links
            if idx > 0:
                diff = min(diff, abs(s_new - old_slots[idx - 1]))
            # Nachbar rechts
            if idx < len(old_slots):
                diff = min(diff, abs(s_new - old_slots[idx]))
            
            # 3. Filterung und Berechnung
            if diff != float('inf'):
                
                # --- WICHTIG: Filter ---
                # Wir bewerten diese Operation NUR, wenn sie in der Nähe eines alten Slots ist.
                # Wenn der Abstand größer als 'ignore_threshold' (z.B. 120 min) ist, 
                # betrachten wir das als "neuen Slot", der nichts mit Stabilität zu tun hat.
                if diff <= ignore_threshold:
                    
                    total_slot_distance += diff
                    total_relevant_ops += 1
                    
                    # Hit Rate (Toleranz 50 Min)
                    if diff <= 50:
                        slot_hits += 1

    # 4. Aggregation
    if total_relevant_ops > 0:
        avg_slot_dist = total_slot_distance / total_relevant_ops
        slot_hit_rate = (slot_hits / total_relevant_ops) * 100
    else:
        avg_slot_dist = 0
        slot_hit_rate = 100.0 # Wenn keine vergleichbaren Slots da waren, ist es "perfekt" (oder 0, je nach Definition)

    return {
        "SlotDev_Avg": avg_slot_dist,
        "SlotHit_Rate": slot_hit_rate
    }