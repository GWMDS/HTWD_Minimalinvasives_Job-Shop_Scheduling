"""
Dataframe Analyses.py contains
- DataFrameChecker
- DataFrameMetricsAnalyser
- DataFramePlotGenerator
"""
import math
from datetime import timedelta

import matplotlib
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Literal, Optional, Union

class DataFrameChecker:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    # Constraints --------------------------------------------------------------------------------------------
    @classmethod
    def check_core_schedule_constraints(
            cls, df_schedule: pd.DataFrame, job_id_column: str = "Job", machine_column: str = "Machine",
            operation_column: str = "Operation", earliest_start_column: str = "Ready Time",
            start_column: str = "Start", end_column: str = "End") -> bool:
        """
        Runs a core consistency check on a production schedule.

        This includes verifying that operations assigned to the same machine do not overlap,
        and that all operations within a job are executed in the correct technological sequence without overlaps.
        """
        checks_passed = True
        if not cls._is_machine_conflict_free(df_schedule, machine_column, start_column, end_column):
            checks_passed = False

        if not cls._is_job_timing_correct(df_schedule, job_id_column, operation_column, start_column, end_column):
            checks_passed = False

        if cls._is_start_correct(df_schedule, start_column, earliest_start_column) is False:
            checks_passed = False

        return checks_passed


    @staticmethod
    def _is_machine_conflict_free(
            df_schedule: pd.DataFrame, machine_column: str = "Machine", start_column: str = "Start",
            end_column: str = "End") -> bool:
        df = df_schedule.sort_values([machine_column, start_column]).reset_index()
        conflict_indices = []

        for machine in df[machine_column].unique():
            machine_df = df[df[machine_column] == machine].sort_values(start_column)

            for i in range(1, len(machine_df)):
                prev = machine_df.iloc[i - 1]
                curr = machine_df.iloc[i]

                if curr[start_column] < prev[end_column]:
                    conflict_indices.extend([prev["index"], curr["index"]])

        conflict_indices = sorted(set(conflict_indices))

        if conflict_indices:
            print(f"- Machine conflicts found: {len(conflict_indices)} rows affected.")
            print(df_schedule.loc[conflict_indices].sort_values([machine_column, start_column]))
            return False
        else:
            print("+ No machine conflicts found.")
            return True

    @classmethod
    def _is_job_timing_correct(
            cls, df_schedule: pd.DataFrame, job_id_column: str = "Job", operation_column: str = "Operation",
            start_column: str = "Start", end_column: str = "End") -> bool:
        violations = []

        for group_id, grp in df_schedule.groupby(job_id_column):
            grp = grp.sort_values(operation_column)
            previous_end = -1
            for _, row in grp.iterrows():
                if row[start_column] < previous_end:
                    violations.append((group_id, int(row[operation_column]), int(row[start_column]), int(previous_end)))
                previous_end = row[end_column]

        if not violations:
            print("+ All job operations are scheduled in non-overlapping, correct sequence.")
            return True

        print(f"- {len(violations)} violation(s) of technological order found:")
        for group_id, op, start, prev_end in violations:
            print(f"  {job_id_column} {group_id!r}, Operation {op}: Start={start}, but previous ended at {prev_end}")

        print("\n> Checking whether the operation sequence by start time matches the technological order:")
        cls._is_operation_sequence_correct(
            df_schedule=df_schedule,
            job_id_column=job_id_column,
            operation_column=operation_column,
            start_column=start_column
        )
        return False

    @staticmethod
    def _is_operation_sequence_correct(
            df_schedule: pd.DataFrame, job_id_column: str = "Job", operation_column: str = "Operation",
            start_column: str = "Start") -> bool:
        violations = []

        for group_id, grp in df_schedule.groupby(job_id_column):
            grp_sorted = grp.sort_values(start_column)
            actual_op_sequence = grp_sorted[operation_column].tolist()
            expected_sequence = sorted(actual_op_sequence)

            if actual_op_sequence != expected_sequence:
                violations.append((group_id, actual_op_sequence))

        if not violations:
            print(f"+ All jobs follow the correct operation sequence.")
            return True
        else:
            print(f"- {len(violations)} job(s) with incorrect order based on {start_column}:")
            for group_id, seq in violations:
                print(f"  {job_id_column} {group_id}: Actual order: {seq}")
            return False


    @staticmethod
    def _is_start_correct(
            df_schedule: pd.DataFrame, job_column: str = "Job", start_column: str = "Start",
            earliest_start_column: str = "Ready Time", verbose: bool = True) -> Optional[bool]:
        required_cols = {job_column, start_column, earliest_start_column}
        missing = required_cols - set(df_schedule.columns)
        if missing:
            if verbose:
                print(f"! Earliest start check not possible: missing column(s): {sorted(missing)}")
            return None

        violations = df_schedule[df_schedule[start_column] < df_schedule[earliest_start_column]]

        if violations.empty:
            if verbose:
                print("+ All operations start at or after the earliest allowed time.")
            return True
        else:
            if verbose:
                print(f"- Invalid early starts found ({len(violations)} row(s)):")
                cols = [job_column, start_column, earliest_start_column]
                print(violations[cols].sort_values(by=start_column))
            return False

    @staticmethod
    def is_duration_correct(
            df_schedule: pd.DataFrame, start_column: str = "Start", end_column: str = "End",
            duration_column: str = "Processing Time") -> bool:
        expected_durations = df_schedule[end_column] - df_schedule[start_column]
        violations = df_schedule[expected_durations != df_schedule[duration_column]]

        if violations.empty:
            print("+ All durations match the difference between start and end.")
            return True
        else:
            print(f"- Duration mismatch found in {len(violations)} row(s):")
            print(violations[[start_column, end_column, duration_column]])
            return False


# Metrics --------------------------------------------------------------------------------------------------------------
class DataFrameMetricsAnalyser:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    @staticmethod
    def get_jobs_metrics_aggregated(
            df_jobs_metrics: pd.DataFrame, column: Literal["Lateness", "Tardiness", "Earliness"] = "Lateness",
            steps: int = 60, min_val: int = -120, max_val: int = 120,
            right_closed: bool = False) -> pd.DataFrame:
        if column not in df_jobs_metrics.columns:
            raise ValueError(f"Column '{column}' does not exist. Available columns: {list(df_jobs_metrics.columns)}")

        if column in ['Tardiness', 'Earliness']:
            min_val = 0
            inner_bins = list(range(min_val, max_val + steps, steps))
            bins = inner_bins + [np.inf]
        else:
            inner_bins = list(range(min_val, max_val + steps, steps))
            if 0 not in inner_bins:
                inner_bins.append(0)
                inner_bins = sorted(inner_bins)
            bins = [-np.inf] + inner_bins + [np.inf]

        zero_count = (df_jobs_metrics[column] == 0).sum()
        non_zero = df_jobs_metrics.loc[df_jobs_metrics[column] != 0, column]

        labels = []
        bin_keys = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            if np.isneginf(lo):
                labels.append(f"<{int(hi)}")
                bin_keys.append(lo + 0.1)
            elif np.isposinf(hi):
                labels.append(f">{int(lo)}")
                bin_keys.append(hi - 0.1)
            else:
                labels.append(f"{int(lo)} - {int(hi)}")
                bin_keys.append((lo + hi) / 2)

        grouped = pd.cut(non_zero, bins=bins, labels=labels, right=right_closed, include_lowest=True)
        counts = grouped.value_counts().reindex(labels, fill_value=0)

        counts["0"] = zero_count
        bin_keys.append(0)
        labels.append("0")

        sort_df = pd.DataFrame({f'{column}_Interval': labels, 'key': bin_keys}).set_index(f'{column}_Interval')
        sorted_counts = counts.loc[sort_df.sort_values('key').index]

        return pd.DataFrame([sorted_counts])


# ----------------------------------------------------------------------------------------------------------------------
class DataFramePlotGenerator:
    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated.")

    # Define base colormap
    tab20 = plt.get_cmap("tab20")

    @classmethod
    def _get_color(cls, idx):
        base_idx = idx % 16
        layer = idx // 16
        if base_idx >= 6:
            base_idx += 1
        rgba = cls.tab20(base_idx / 20)
        r, g, b, _ = rgba
        if layer == 1:
            r, g, b = max(0.0, r * 0.9), min(1.0, g * 1.4), max(0.0, b * 0.9)
        elif layer == 2:
            r, g, b = min(1.0, r * 1.15), max(0.0, g * 0.85), min(1.0, b * 1.15)

        return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'

    # Check due date ------------------------------------------------------------------------------------------------
    @staticmethod
    def get_elapsed_time_density_plot_figure(
            df_times: pd.DataFrame, routing_column: str = "Routing_ID", earliest_start_column: str = "Ready Time",
            simulated_end_column: str = "End", total_duration_column: str = "Total Processing Time",
            bins: int = 30, y_max: float = 0.002):
        routings = df_times[routing_column].unique()
        n_routings = len(routings)
        n_cols = min(4, n_routings)
        n_rows = int(np.ceil(n_routings / n_cols))

        elapsed_times = df_times[simulated_end_column] - df_times[earliest_start_column]
        x_max = elapsed_times.max()

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes = axes.ravel()

        for idx, routing in enumerate(routings):
            ax: Axes = axes[idx]  # type: ignore
            dfr = df_times[df_times[routing_column] == routing]

            avg_elapsed_times = dfr[simulated_end_column] - dfr[earliest_start_column]
            avg_elapsed_time = avg_elapsed_times.mean()

            duration = dfr[total_duration_column].mean()

            sns.histplot(
                avg_elapsed_times, bins=bins, kde=True, stat="density",
                ax=ax, color="cornflowerblue", edgecolor="black"
            )

            ax.axvline(duration, color="gray", linestyle="--", label="Duration")
            ax.axvline(avg_elapsed_time, color='orange', linestyle='--', label="Avg. elapsed time")

            ax.set_title(f'Routing {routing}')
            ax.set_xlabel("Elapsed time [min]\n(simulated end - earliest start)")
            ax.set_ylabel('Density [1/min]')
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.legend()

        for j in range(n_routings, len(axes)):
            fig.delaxes(axes[j])  # type: ignore

        plt.tight_layout()
        return fig

    @staticmethod
    def get_elapsed_time_density_plot_figure_side_by_side(
            df_times: pd.DataFrame,
            routing_filter: list[str] | None = None,
            routing_column: str = "Routing_ID",
            earliest_start_column: str = "Ready Time",
            simulated_end_column: str = "End",
            total_duration_column: str = "Total Processing Time",
            bins: int = 30,
            y_max: float = 0.002,
            x_max: float | None = None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd

        if routing_filter is not None:
            df_times = df_times[df_times[routing_column].isin(routing_filter)]

        routings = df_times[routing_column].unique()
        n_routings = len(routings)

        fig, axes = plt.subplots(1, n_routings, figsize=(5 * n_routings, 4), sharey=True)
        if n_routings == 1:
            axes = [axes]

        elapsed_times_all = df_times[simulated_end_column] - df_times[earliest_start_column]
        global_y_max = y_max or (0.002 * (n_routings / 2))
        if x_max is None:
            x_max = float(elapsed_times_all.max() * 1.1)

        for idx, routing in enumerate(routings):
            ax = axes[idx]
            dfr = df_times[df_times[routing_column] == routing]

            elapsed_times = dfr[simulated_end_column] - dfr[earliest_start_column]
            avg_elapsed_time = elapsed_times.mean()
            duration = dfr[total_duration_column].mean()

            sns.histplot(
                elapsed_times, bins=bins, kde=True, stat="density",
                ax=ax, color="cornflowerblue", edgecolor="black"
            )

            ax.axvline(duration, color="gray", linestyle="--", label="Σ op duration")
            ax.axvline(avg_elapsed_time, color="orange", linestyle="--", label="Ø elapsed time")

            ax.set_title(f"Routing {routing}")

            if idx == 0:
                ax.set_ylabel("Density [1/min]")
                ax.tick_params(axis="y", left=True, labelleft=True)
                ax.spines["left"].set_visible(True)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.spines["left"].set_visible(True)

            if idx == n_routings - 1:
                ax.set_xlabel("Timespan from earliest start [min]", ha="right", x=1.0)
                ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
            else:
                ax.set_xlabel("")

            ax.set_xlim(0, x_max)
            ax.set_ylim(0, global_y_max)

        fig.tight_layout()
        return fig



    @staticmethod
    def get_scheduling_window_density_plot_figure(
            df_times: pd.DataFrame, routing_column: str = "Routing_ID",
            earliest_start_column: str = "Ready Time", due_date_column: str = "Due Date",
            total_duration_column: str = "Total Processing Time", bins: int = 30, y_max: float = 0.002):
        routings = df_times[routing_column].unique()
        n_routings = len(routings)
        n_cols = min(4, n_routings)
        n_rows = int(np.ceil(n_routings / n_cols))

        all_scheduling_windows = df_times[due_date_column] - df_times[earliest_start_column]
        x_max = all_scheduling_windows.max()

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes = axes.ravel()

        for idx, routing in enumerate(routings):
            ax: Axes = axes[idx]  # type: ignore
            dfr = df_times[df_times[routing_column] == routing]

            scheduling_windows = dfr[due_date_column] - dfr[earliest_start_column]
            avg_scheduling_window = scheduling_windows.mean()

            duration = dfr[total_duration_column].mean()

            sns.histplot(
                scheduling_windows, bins=bins, kde=True, stat="density",
                ax=ax, color="cornflowerblue", edgecolor="black"
            )

            ax.axvline(duration, color="gray", linestyle="--", label="Duration")
            ax.axvline(avg_scheduling_window, color="orange", linestyle="--", label="Avg. scheduling window")

            ax.set_title(f'Routing {routing}')
            ax.set_xlabel("Scheduling time window [min]\n(due date - earliest start)")
            ax.set_ylabel('Density [1/min]')
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.legend()

        for j in range(n_routings, len(axes)):
            fig.delaxes(axes[j])  # type: ignore

        plt.tight_layout()
        return fig

    @staticmethod
    def get_scheduling_window_density_plot_figure_side_by_side(
            df_times: pd.DataFrame,
            routing_filter: list[str] | None = None,
            routing_column: str = "Routing_ID",
            earliest_start_column: str = "Ready Time",
            due_date_column: str = "Due Date",
            total_duration_column: str = "Total Processing Time",
            bins: int = 30,
            y_max: float = 0.002,
            x_max: float | None = None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd

        if routing_filter is not None:
            df_times = df_times[df_times[routing_column].isin(routing_filter)]

        routings = df_times[routing_column].unique()
        n_routings = len(routings)

        fig, axes = plt.subplots(1, n_routings, figsize=(5 * n_routings, 4), sharey=True)
        if n_routings == 1:
            axes = [axes]

        all_scheduling_windows = df_times[due_date_column] - df_times[earliest_start_column]
        global_y_max = y_max or (0.002 * (n_routings / 2))
        if x_max is None:
            x_max = float(all_scheduling_windows.max() * 1.1)

        for idx, routing in enumerate(routings):
            ax = axes[idx]
            dfr = df_times[df_times[routing_column] == routing]

            scheduling_windows = dfr[due_date_column] - dfr[earliest_start_column]
            avg_scheduling_window = scheduling_windows.mean()
            duration = dfr[total_duration_column].mean()

            sns.histplot(
                scheduling_windows, bins=bins, kde=True, stat="density",
                ax=ax, color="cornflowerblue", edgecolor="black"
            )

            ax.axvline(duration, color="gray", linestyle="--", label="Σ op duration")
            ax.axvline(avg_scheduling_window, color="orange", linestyle="--", label="Ø scheduling window")
            ax.set_title(f"Routing {routing}")

            if idx == 0:
                ax.set_ylabel("Density [1/min]")
                ax.tick_params(axis="y", which="both", left=True, labelleft=True)
                ax.spines["left"].set_visible(True)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                ax.spines["left"].set_visible(True)

            if idx == n_routings - 1:
                ax.set_xlabel("Timespan from earliest start [min]", ha="right", x=1.0)
                ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
            else:
                ax.set_xlabel("")

            ax.set_xlim(0, x_max)
            ax.set_ylim(0, global_y_max)

        fig.tight_layout()
        return fig

    # Gantt chart for schedule or simulation -------------------------------------------------------------------------
    @classmethod
    def get_gantt_chart_figure(
            cls, df_workflow: pd.DataFrame, title: str = "Gantt chart",
            job_column: str = "Job", machine_column: str = "Machine", duration_column: str = "Processing Time",
            perspective: Literal["Machine", "Job"] = "Machine",
            # --- NEU: Optionaler DataFrame für Referenz-Linien ---
            reference_df: Optional[pd.DataFrame] = None,
            start_column: str = "Start"
            ):
        """
        Create a Gantt chart figure from either a job or machine perspective.
        Optional: Draws red lines for start times from reference_df (e.g. previous shift).

        :param df_workflow: DataFrame containing scheduling or simulation data
        :param title: Title of the chart
        :param job_column: Column name identifying jobs
        :param machine_column: Column name identifying machines
        :param duration_column: Column name for operation durations
        :param perspective: Either "Job" (job-centric view) or "Machine" (machine-centric view)
        :param reference_df: Optional DataFrame with reference start times (e.g. from previous shift)
        :param start_column: Column name for start times
        :return: Matplotlib Figure object
        """

        # Axis and color settings
        if perspective == "Job":
            group_column = job_column
            color_column = machine_column
        elif  perspective == "Machine":
            group_column = machine_column
            color_column = job_column
        else:
            raise ValueError("Perspective must be 'Job' or 'Machine'")
        y_label = group_column

        groups = sorted(df_workflow[group_column].unique())
        color_items = sorted(df_workflow[color_column].unique())
        y_ticks = range(len(groups))
        
        # Mapping von Gruppen-Namen zu Y-Achsen-Index für Referenzlinien
        group_to_y_idx = {g: i for i, g in enumerate(groups)}

        color_map = {item: cls._get_color(i) for i, item in enumerate(color_items)}

        fig_height = len(groups) * 0.8
        fig, ax = plt.subplots(figsize=(16, fig_height))

        # 1. Haupt-Balken zeichnen
        for idx, group in enumerate(groups):
            rows = df_workflow[df_workflow[group_column] == group]
            for _, row in rows.iterrows():
                ax.barh(idx,
                        row[duration_column],
                        left=row[start_column],
                        height=0.5,
                        color=color_map[row[color_column]],
                        edgecolor='black',
                        zorder=2) # zorder ensures bars are above grid

        # 2. NEU: Referenz-Linien (Vorgänger-Shift) zeichnen
        if reference_df is not None and not reference_df.empty:
            for _, row in reference_df.iterrows():
                group_val = row[group_column] # z.B. "M01" (wenn perspective="Machine")
                
                # Nur zeichnen, wenn die Maschine im aktuellen Plot existiert
                if group_val in group_to_y_idx:
                    y_pos = group_to_y_idx[group_val]
                    x_pos = row[start_column]
                    
                    # Zeichne rote vertikale Linie über die Höhe des Balkens
                    ax.plot([x_pos, x_pos], [y_pos - 0.4, y_pos + 0.4], 
                            color='red', linewidth=2, linestyle='-', zorder=3, alpha=0.8)

        # Legend
        legend_handles = [mpatches.Patch(color=color_map[item], label=str(item)) for item in color_items]
        
        # Füge Legende für rote Linie hinzu, falls Referenzdaten da sind
        if reference_df is not None:
             legend_handles.append(plt.Line2D([0], [0], color='red', linewidth=2, label='Prev. Shift Start'))

        legend_columns = (len(color_items) // 35) + 1
        ax.legend(handles=legend_handles,
                  title=color_column,
                  bbox_to_anchor=(1.01, 1),
                  loc='upper left',
                  ncol=legend_columns,
                  handlelength=2.4,
                  frameon=False,
                  alignment='left'
                  )

        # Axis labels and formatting
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(groups)
        ax.set_xlabel("Time (in minutes)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)

        # Time axis scaling
        max_time = (df_workflow[start_column] + df_workflow[duration_column]).max()
        x_start = int((df_workflow[start_column].min() // 1440) * 1440)
        ax.set_xlim(x_start, max_time + 60)

        x_ticks = list(range(x_start, int(max_time) + 360, 360))
        ax.set_xticks(x_ticks)
        ax.grid(True, axis='x', linestyle='--', alpha=0.6, zorder=0)

        # Vertical lines every 1440 minutes (e.g., day delimiter)
        for x in range(x_start, int(max_time) + 1440, 1440):
            ax.axvline(x=x, color='#777777', linestyle='-', linewidth=1.0, alpha=0.7)

        plt.tight_layout()
        return fig

    # Convergence of Solver ------------------------------------------------------------------------------------------
    @staticmethod
    def _format_hhmm(seconds: Union[float, int, np.signedinteger]) -> str:
        """Format seconds as HH:MM (no seconds)."""
        td = timedelta(seconds=int(seconds))
        total_minutes = int(td.total_seconds() // 60)
        h, m = divmod(total_minutes, 60)
        return f"{h:02d}:{m:02d}"

    @staticmethod
    def _format_h(seconds: Union[float, int, np.signedinteger]) -> str:
        """Format seconds as H (nur Stunden)."""
        return f"{int(int(seconds) // 3600)}"

    @staticmethod
    def _choose_granularity(
            x_max_s: float, granularity: Literal["auto", "seconds", "minutes", "quarter", "half", "hours"]) -> str:
        if granularity != "auto":
            return granularity
        if x_max_s <= 5 * 60:
            return "seconds"
        if x_max_s <= 30 * 60:
            return "minutes"
        if x_max_s <= 60 * 60:
            return "quarter"
        if x_max_s <= 5 * 3600:
            return "half"
        return "hours"

    @staticmethod
    def _step_for_granularity(x_max: float, gran: str) -> int:
        if gran == "seconds":
            if x_max <= 10:
                return 1
            elif x_max <= 50:
                return 5
            else:
                return 10
        if gran == "minutes":
            if x_max <= 15 * 60:  # bis 15 Minuten
                return 60  # 1 Minute
            elif x_max <= 90 * 60:  # bis 90 Minuten
                return 300  # 5 Minuten
            else:
                return 600  # 10 Minuten
        if gran == "quarter":
            return 15 * 60
        if gran == "half":
            return 30 * 60
        if gran == "hours":
            return 3600
        return 60  # Fallback: 1 minute

    @classmethod
    def get_convergence_plot_figure(
            cls, df: pd.DataFrame, time_col: str = "Time", bestsol_col: str = "BestSol", title: Optional[str] = None,
            y_min: float = None, y_max: float = None, x_max: Optional[float] = None, df_max_time: float = None,
            granularity: Literal["auto", "seconds", "minutes", "quarter", "half", "hours"] = "auto", marker: str = "."):
        """
        Generate a convergence curve plot of the solver's best solution over time.

        :param df: DataFrame containing solver log data.
        :param time_col: Column name for elapsed time values.
        :param bestsol_col: Column name for best solution values.
        :param title: Optional title.
        :param y_min: Minimum y-axis value; if None, uses the minimum in `bestsol_col`.
        :param y_max: Maximum y-axis value; if None, uses the maximum in `bestsol_col`.
        :param df_max_time: Maximum time (in seconds) to display on the x-axis.
        :param granularity: Tick label granularity for the x-axis; "auto", "seconds", "minutes", "quarter", "half", or "hours".
        :param marker: Matplotlib marker style for data points.
        :return: Matplotlib Figure object, or None if the filtered DataFrame is empty.
        :rtype: matplotlib.figure.Figure | None
        """
        if y_min is not None and y_max is not None and y_max < y_min:
            return None

        d = df.copy()
        if df_max_time is not None:
            d = d[d[time_col] <= df_max_time]

        if d.empty:
            print("DataFrame hat keine Zeilen im angegebenen Zeitbereich.")
            return None

        # x-axis
        x_max = x_max if x_max else float(d[time_col].max())
        gran = cls._choose_granularity(x_max, granularity)
        step = cls._step_for_granularity(x_max, gran)
        ticks_s = np.arange(0, x_max +0.001, step)

        if gran == "seconds":
            tick_labels = [f"{int(s)}" for s in ticks_s]
            xlabel = "Zeit [sec]"
        elif gran == "hours":
            # nur Stunden, 00, 01, 02, ...
            tick_labels = [cls._format_h(s) for s in ticks_s]
            xlabel = "Zeit [h]"
        else:
            tick_labels = [cls._format_hhmm(s) for s in ticks_s]
            xlabel = "Zeit [hh:mm]"

        # y axis
        y_step = 60
        ymin = y_min if y_min is not None else float(d[bestsol_col].min())
        ymax = y_max if y_max is not None else float(d[bestsol_col].max())
        raw = (ymax - ymin) / 8
        pow10 = 10 ** int(np.floor(np.log10(raw))) if raw > 0 else 1
        for m in (1, 2, 5, 10):
            if raw <= m * pow10:
                y_step = m * pow10
                break

        # Round ymin to multiples of y_step
        ymin = np.floor(ymin / y_step) * y_step

        # Matplot figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(d[time_col], d[bestsol_col], marker=marker)
        ax.set_xlabel(xlabel)
        ax.set_xticks(ticks_s)
        ax.set_xticklabels(tick_labels)

        ax.set_ylabel("Best Solution")
        if y_step:
            ymax = round_up_to_multiple(ymax, y_step)
            yticks = np.arange(ymin, ymax+0.002, y_step)
            ax.set_yticks(yticks)
        ax.set_ylim(ymin, ymax)
        ax.grid(True)
        if title: # "Convergence Curve of the OR-Tools CP-SAT Solver"
            ax.set_title(title)

            # Spines sichtbar
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)

        if y_max:
            ax.set_ylim(top=y_max)

        # Spines anpassen: linestyle und alpha
        for side in ["top", "right"]:
            ax.spines[side].set_linestyle("--")
            ax.spines[side].set_alpha(0.14)
            ax.spines[side].set_linewidth(0.8)


        if step >= 300:
            ax.set_xlim(-10, x_max)
        else:
            ax.set_xlim(0, x_max)

        fig.tight_layout()
        return fig

    @classmethod
    def get_relative_convergence_plot_figure(
        cls,
        df: pd.DataFrame,
        time_col: str = "Time",
        bestsol_col: str = "BestSol",
        subtitle: str = "",
        max_time: float | None = None,
        granularity: Literal["auto", "seconds", "minutes", "quarter", "half", "hours"] = "auto",
        marker: str = ".",
    ):
        """
        Generate a convergence curve with a relative y-axis (first value = 0%, last value = 100%).

        :param df: DataFrame containing solver log data.
        :param time_col: Column name for elapsed time values.
        :param bestsol_col: Column name for best solution values.
        :param subtitle: Optional subtitle to append to the plot title.
        :param max_time: Maximum time (in seconds) to display on the x-axis.
        :param granularity: Tick label granularity for the x-axis; "auto", "seconds", "minutes", "quarter", "half", or "hours".
        :param marker: Matplotlib marker style for data points.
        :return: Matplotlib Figure object, or None if the filtered DataFrame is empty.
        :rtype: matplotlib.figure.Figure | None
        """
        d = df.copy()

        # Filter by max_time if provided
        if max_time is not None:
            d = d[d[time_col] <= max_time]

        if d.empty:
            print("DataFrame has no rows in the given time range.")
            return None

        # Normalize Y: first -> 0%, last -> 100%
        first_val = float(d[bestsol_col].iloc[0])
        last_val  = float(d[bestsol_col].iloc[-1])
        if first_val == last_val:
            d[bestsol_col] = 100.0
        else:
            d[bestsol_col] = (d[bestsol_col] - first_val) / (last_val - first_val) * 100.0

        # X-axis preparation
        x_max = float(max_time) if max_time is not None else float(d[time_col].max())
        gran = cls._choose_granularity(x_max, granularity)
        step = cls._step_for_granularity(x_max, gran)
        ticks_s = np.arange(0, int(x_max) + step, step)

        if gran == "seconds":
            tick_labels = [f"{int(s)}" for s in ticks_s]
            xlabel = "Time [sec]"
        else:
            tick_labels = [cls._format_hhmm(s) for s in ticks_s]
            xlabel = "Time [hh:mm]"

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(d[time_col], d[bestsol_col], marker=marker)

        ax.set_xlabel(xlabel)
        ax.set_xticks(ticks_s)
        ax.set_xticklabels(tick_labels)

        ax.set_ylabel("Relative Best Solution [%]")
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))

        ax.grid(True)
        title = "Convergence Curve of the OR-Tools CP-SAT Solver"
        if subtitle:
            title += f" – {subtitle}"
        ax.set_title(title)

        fig.tight_layout()
        return fig

def round_up_to_multiple(a, b):
    if b == 0:
        raise ValueError("b darf nicht 0 sein.")
    return math.ceil(a / b) * b