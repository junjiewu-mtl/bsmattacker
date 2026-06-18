"""
False Deceleration Cruising Attack
==================================

This attack targets vehicles in CRUISING scenarios (steady-state driving).
The attacker falsely reports sudden deceleration, potentially causing following
vehicles to brake unnecessarily and creating traffic disruption or rear-end collisions.

Attack Mechanism:
- Target: Messages with Scenario_Label = 'Cruising'
- Modification: Gradual speed reduction with consistent deceleration profile
- Realism: Physics-consistent speed/acceleration over consecutive messages
- Co-spoofing: Updates position using the kinematic equation
  x = x₀ + v·dt + ½a·dt² so the GPS trajectory matches the reported braking,
  avoiding a position-velocity consistency anomaly.

Literature Reference:
- VeReMi "Speed Falsification" attacks (Constant/Random Speed)
- Key improvement: Physics-consistent braking profile over time

Use Case:
This attack exploits the trust in BSM data for cooperative adaptive cruise control (CACC)
and collision warning systems, potentially causing cascading braking events.
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class FalseDecelerationCruisingAttacker(BaseAttacker):
    """
    Implements the 'False Deceleration Cruising' attack for cruising scenarios.

    Physics-consistent braking with co-spoofed position deceleration.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="False Deceleration Cruising", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                     attack_ratio: float = 0.2,
                     target_decel: tuple = (-5.0, -4.0),
                     min_brake_duration_msgs: int = 3,
                     physics_consistent: bool = True,
                     per_frame_noise_std: float = 0.5,
                     onset_ramp_msgs: int = 1,
                     taper_ramp_msgs: int = 2,
                     target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'False Deceleration Cruising' attacks into cruising messages.

        **Target safety app: EEBL** (Emergency Electronic Brake Light).
        Per SAE J2945/1, EEBL fires on sustained ``accel_long ≤ -3.92 m/s²``
        (0.4 g) for ≥ 200 ms (≈2 consecutive BSMs at 10 Hz). The sustain-phase
        decel is held below that threshold for at least 2 frames so the attack
        trips the target app.

        Position is co-spoofed using the kinematic equation
        x = x₀ + v·dt + ½a·dt² so the GPS trajectory matches the reported
        braking. The brake is decomposed into three phases so only the sustain
        frames carry the EEBL-tripping deep decel, while onset + taper frames
        sit closer to the benign acceleration distribution:
          * ``onset_ramp_msgs`` frame(s) ramp 0 → sustain (builds into the
            EEBL threshold; these frames are in benign accel range).
          * Sustain frames hold accel in ``target_decel`` (default (-5, -4)
            m/s², unambiguously below -3.92 so EEBL trips for ≥ 200 ms).
          * ``taper_ramp_msgs`` frames ramp sustain → ~0 (back toward
            benign accel values).
          * Per-frame Gaussian noise (``per_frame_noise_std``) breaks the
            constant-value signature so any single-frame read still varies.
          * Speed integration respects the varying accel (sums
            decel·dt rather than using a single fake_decel·elapsed_time).

        Args:
            df: Input DataFrame with BSM data (must have 'Scenario_Label' column)
            attack_ratio: Proportion of cruising messages to attack (0.0 to 1.0)
            target_decel: (min, max) sustain-phase decel in m/s². Default
                (-3.0, -1.5) matches the benign accel_long tail so the attack
                is realistic rather than trivially flagged.
            min_brake_duration_msgs: Minimum consecutive messages for braking event
            physics_consistent: If True, speed decreases gradually with consistent
                per-frame deceleration (integrated sum, not closed-form).
            per_frame_noise_std: Gaussian std (m/s²) added to each frame's decel
                to break constant-value signature. Default 0.3.
            onset_ramp_msgs: Number of leading frames used for onset ramp
                (accel eases from 0 into the target range).
            taper_ramp_msgs: Number of trailing frames used for taper
                (accel eases from target back toward 0).
            target_vehicles: List of Device_IDs to attack (None = attack all vehicles)

        Returns:
            DataFrame with injected attacks
        """
        df = df.copy()

        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        time_col = get_column_name(df, 'timestamp') or 'Tx_Timestamp'
        lat_col = get_column_name(df, 'latitude') or 'Latitude_deg'
        lon_col = get_column_name(df, 'longitude') or 'Longitude_deg'
        heading_col = get_column_name(df, 'heading') or 'Heading_deg'
        speed_col = get_column_name(df, 'speed') or 'Speed_mps'
        accel_col = get_column_name(df, 'accel_long') or 'Accel_Long_mps2'
        scenario_col = get_column_name(df, 'scenario_label') or 'Scenario_Label'

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        if scenario_col not in df.columns:
            raise ValueError(f"DataFrame must have '{scenario_col}' column. Run scenario labeling first.")

        attack_mask = pd.Series(False, index=df.index)

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        total_cruising = 0
        total_attacked = 0

        for vehicle_id in target_vehicles:
            vehicle_mask = (df[device_col] == vehicle_id) & (df[scenario_col] == 'Cruising')
            vehicle_cruising_indices = df[vehicle_mask].index.tolist()

            if len(vehicle_cruising_indices) == 0:
                continue

            total_cruising += len(vehicle_cruising_indices)
            segments = self._find_contiguous_segments(vehicle_cruising_indices)
            valid_segments = [s for s in segments if len(s) >= min_brake_duration_msgs]
            if not valid_segments:
                valid_segments = segments

            total_vehicle_cruising = len(vehicle_cruising_indices)
            num_to_attack = max(1, int(total_vehicle_cruising * attack_ratio))

            attacked_so_far = 0
            for segment in valid_segments:
                if attacked_so_far >= num_to_attack:
                    break

                segment_attack_count = min(
                    len(segment),
                    max(min_brake_duration_msgs, int(len(segment) * attack_ratio)),
                    num_to_attack - attacked_so_far
                )

                attack_indices = segment[:segment_attack_count]
                attacked_so_far += len(attack_indices)
                total_attacked += len(attack_indices)

                # v3.0: sustain-phase deep decel for EEBL +
                # onset/taper frames carry the ORIGINAL cruising-frame accel
                # values (resampled from the preceding benign window for this
                # vehicle). By leaving onset/taper accel values in the true
                # benign distribution, only the sustain frames (guaranteed
                # ≥ 2) carry the EEBL-tripping decel signature. Per-frame
                # audit on accel_long becomes harder: majority of attacker-
                # labeled frames are drawn from the benign distribution by
                # construction.
                sustain_decel = np.random.uniform(target_decel[0], target_decel[1])
                n_frames = len(attack_indices)

                # Dynamic onset/taper sizing: reserve >= 2 frames for sustain
                max_ramps = max(0, n_frames - 2)
                onset = min(onset_ramp_msgs, max_ramps // 2)
                taper = min(taper_ramp_msgs, max_ramps - onset)
                sustain_frames = n_frames - onset - taper
                # short segments that can't hold >=2 sustain frames
                # (after onset+taper reserve) get skipped rather than aborting the
                # whole inject. ASCII-only error message prevents Windows joblib
                # loky cp1252 encoding hang when workers pickle the exception.
                if sustain_frames < 2:
                    continue  # try next segment; this one is too short for EEBL
                assert sustain_frames >= 2, (
                    f"brake_bluff v3: sustain_frames={sustain_frames} must be "
                    f">= 2 to trip EEBL; increase min_brake_duration_msgs."
                )

                # Define initial_idx early — needed by the benign-pool sampler
                # below and by the speed/position integrators further down.
                initial_idx = attack_indices[0]
                # Sample a pool of real benign accel values from this
                # vehicle's preceding cruising window for onset + taper.
                preceding_mask = (
                    (df[device_col] == vehicle_id)
                    & (df[scenario_col] == 'Cruising')
                    & (df.index < initial_idx)
                )
                preceding_accel_pool = df.loc[preceding_mask, accel_col].dropna().to_numpy()
                # Fallback to global cruising accel if this vehicle has no history
                if len(preceding_accel_pool) < (onset + taper + 5):
                    global_cruising = df.loc[
                        df[scenario_col] == 'Cruising', accel_col
                    ].dropna().to_numpy()
                    preceding_accel_pool = (
                        global_cruising if len(global_cruising) > 0
                        else np.array([0.0])
                    )

                decel_profile = np.full(n_frames, sustain_decel)
                if onset > 0:
                    # Onset frames: resample real benign accel values
                    decel_profile[:onset] = np.random.choice(
                        preceding_accel_pool, size=onset, replace=True
                    )
                if taper > 0:
                    # Taper frames: resample real benign accel values
                    decel_profile[-taper:] = np.random.choice(
                        preceding_accel_pool, size=taper, replace=True
                    )
                # Small per-frame noise on sustain frames only to break any
                # constant-value signature within the sustain window.
                sustain_slice = slice(onset, onset + sustain_frames)
                decel_profile[sustain_slice] = (
                    decel_profile[sustain_slice]
                    + np.random.normal(0.0, per_frame_noise_std, size=sustain_frames)
                )
                # Clamp so noise cannot flip sign or escape plausible bounds
                decel_profile = np.clip(decel_profile, -8.0, 2.0)

                # initial_idx already set above (moved there for preceding-pool sampling)
                initial_speed = df.loc[initial_idx, speed_col]

                dt = 0.1
                if len(attack_indices) > 1:
                    try:
                        t0 = pd.to_numeric(df.loc[attack_indices[0], time_col], errors='coerce')
                        t1 = pd.to_numeric(df.loc[attack_indices[1], time_col], errors='coerce')
                        if pd.notna(t0) and pd.notna(t1) and abs(t1 - t0) > 0:
                            dt = abs(t1 - t0)
                    except (ValueError, TypeError):
                        pass

                heading_rad = np.radians(df.loc[initial_idx, heading_col])
                cum_displacement = 0.0
                speed_series = np.empty(n_frames)
                speed_series[0] = initial_speed

                for i, idx in enumerate(attack_indices):
                    frame_decel = float(decel_profile[i])

                    if physics_consistent:
                        if i == 0:
                            new_speed = initial_speed
                        else:
                            # Integrate varying accel (not closed-form)
                            new_speed = max(0.0, speed_series[i - 1] + frame_decel * dt)
                        speed_series[i] = new_speed
                    else:
                        reduction_ratio = np.random.uniform(0.5, 0.8)
                        new_speed = initial_speed * (1 - reduction_ratio)

                    df.loc[idx, speed_col] = new_speed
                    df.loc[idx, accel_col] = frame_decel

                    # v2.0+: co-spoof position along heading, using integrated speed
                    if i > 0 and physics_consistent:
                        step_disp = speed_series[i - 1] * dt + 0.5 * frame_decel * dt ** 2
                        step_disp = max(0.0, step_disp)
                        cum_displacement += step_disp

                        # Real-position displacement vs spoofed displacement
                        real_cruise_disp = initial_speed * i * dt
                        backward_offset = real_cruise_disp - cum_displacement
                        off_n = -backward_offset * np.cos(heading_rad)
                        off_e = -backward_offset * np.sin(heading_rad)

                        lat = df.loc[idx, lat_col]
                        lon = df.loc[idx, lon_col]
                        new_lat, new_lon = self.offset_coordinates(lat, lon, off_n, off_e)
                        df.loc[idx, lat_col] = new_lat
                        df.loc[idx, lon_col] = new_lon

                    attack_mask.loc[idx] = True

        if total_cruising == 0:
            print(f"Warning: No cruising messages found to attack. Returning original DataFrame.")
            return self.add_attack_labels(df, attack_mask, self.attack_name)

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        self.log_attack_summary(df, attack_mask)

        return df
    
    def _find_contiguous_segments(self, indices: list) -> list:
        """
        Find contiguous segments in a list of indices.
        
        Returns list of lists, where each inner list is a contiguous segment.
        """
        if not indices:
            return []
        
        segments = []
        current_segment = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_segment.append(indices[i])
            else:
                segments.append(current_segment)
                current_segment = [indices[i]]
        
        segments.append(current_segment)
        return segments
