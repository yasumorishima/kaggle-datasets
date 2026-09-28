(function() {
    const columns = {
        "pitch_type": "Pitch type code as classified by Statcast (FF four-seam, SI sinker, FC cutter, SL slider, ST sweeper, CU curveball, CH changeup, FS splitter, ...).",
        "game_date": "Date of the game (YYYY-MM-DD format)",
        "release_speed": "Pitch velocity at release point in miles per hour (mph)",
        "release_pos_x": "Horizontal release point in feet (catcher's perspective: negative=left, positive=right)",
        "release_pos_z": "Vertical release point in feet (height above ground)",
        "player_name": "The pitcher's name as Baseball Savant writes it ('Last, First'). The build checks it against MLB StatsAPI's name for the pitcher id.",
        "player_name_eng": "Added by this dataset (not a Statcast column): the pitcher's full name from MLB StatsAPI ('First Last').",
        "batter": "MLBAM id of the batter facing the pitch.",
        "pitcher": "MLBAM id of the pitcher: the Japanese player of the row. Joins to mlbam_id in players.csv and season_summary.csv (role pitcher).",
        "events": "Plate-appearance outcome (e.g. single, strikeout, home_run, walk); filled only on the last pitch of a plate appearance.",
        "description": "Individual pitch result (ball, called_strike, swinging_strike, hit_into_play, foul, etc.)",
        "spin_dir": "Deprecated Savant column, kept so the files have Savant's full schema; empty in every row (the build fails otherwise).",
        "spin_rate_deprecated": "Deprecated Savant column, kept so the files have Savant's full schema; empty in every row (the build fails otherwise).",
        "break_angle_deprecated": "Deprecated Savant column, kept so the files have Savant's full schema; empty in every row (the build fails otherwise).",
        "break_length_deprecated": "Deprecated Savant column, kept so the files have Savant's full schema; empty in every row (the build fails otherwise).",
        "zone": "Strike zone location (1-9=strike zone grid, 11-14=outside zone)",
        "des": "Detailed text description of the play",
        "game_type": "Game type. Always R (regular season) in this dataset; spring training and postseason pitches are not included.",
        "stand": "Batter's stance (R=right-handed, L=left-handed)",
        "p_throws": "Pitcher's throwing hand (R=right, L=left)",
        "home_team": "Home team abbreviation (e.g., NYY, LAD)",
        "away_team": "Away team abbreviation",
        "type": "Pitch outcome category (S=strike, B=ball, X=in play)",
        "hit_location": "Fielder position number where ball was hit (1=pitcher, 2=catcher, 3=1B, etc.)",
        "bb_type": "Batted ball type (ground_ball, line_drive, fly_ball, popup)",
        "balls": "Number of balls in the count (0-3)",
        "strikes": "Number of strikes in the count (0-2)",
        "game_year": "Season of the game (2015-2026).",
        "pfx_x": "Horizontal movement in feet from the catcher's perspective, without gravity (Statcast's induced movement).",
        "pfx_z": "Vertical movement in feet without gravity (induced vertical break): positive = the pitch stays above the path of a spinless pitch.",
        "plate_x": "Horizontal position at home plate in feet (catcher's perspective: negative=left, positive=right)",
        "plate_z": "Vertical position at home plate in feet (height above ground)",
        "on_3b": "MLBAM player ID of runner on 3rd base (NaN if empty)",
        "on_2b": "MLBAM player ID of runner on 2nd base (NaN if empty)",
        "on_1b": "MLBAM player ID of runner on 1st base (NaN if empty)",
        "outs_when_up": "Number of outs when batter came to plate (0-2)",
        "inning": "Inning number",
        "inning_topbot": "Top or Bottom of inning",
        "hc_x": "Hit coordinate X position (pixel coordinates on field diagram)",
        "hc_y": "Hit coordinate Y position (pixel coordinates on field diagram)",
        "tfs_deprecated": "Deprecated Savant column, kept so the files have Savant's full schema; empty in every row (the build fails otherwise).",
        "tfs_zulu_deprecated": "Deprecated Savant column, kept so the files have Savant's full schema; empty in every row (the build fails otherwise).",
        "umpire": "Deprecated Savant column, kept so the files have Savant's full schema; empty in every row (the build fails otherwise).",
        "sv_id": "Savant's old pitch id (YYMMDD_hhmmss); filled in 2015-2016 only in this dataset.",
        "vx0": "Initial velocity in X direction (feet per second) at y=50 feet",
        "vy0": "Initial velocity in Y direction (feet per second) at y=50 feet",
        "vz0": "Initial velocity in Z direction (feet per second) at y=50 feet",
        "ax": "Acceleration in X direction (feet per second squared)",
        "ay": "Acceleration in Y direction (feet per second squared)",
        "az": "Acceleration in Z direction (feet per second squared)",
        "sz_top": "Top of batter's strike zone in feet (varies by batter height)",
        "sz_bot": "Bottom of batter's strike zone in feet (varies by batter height)",
        "hit_distance_sc": "Projected hit distance in feet (Statcast calculation)",
        "launch_speed": "Exit velocity off the bat in miles per hour (mph)",
        "launch_angle": "Vertical launch angle in degrees (negative=ground ball, positive=fly ball)",
        "effective_speed": "Perceived velocity adjusted for release point extension (mph)",
        "release_spin_rate": "Spin rate at release in revolutions per minute (RPM)",
        "release_extension": "Release point extension toward home plate in feet (distance from rubber)",
        "game_pk": "Unique game identifier in MLB's database",
        "fielder_2": "MLBAM player ID of catcher",
        "fielder_3": "MLBAM player ID of first baseman",
        "fielder_4": "MLBAM player ID of second baseman",
        "fielder_5": "MLBAM player ID of third baseman",
        "fielder_6": "MLBAM player ID of shortstop",
        "fielder_7": "MLBAM player ID of left fielder",
        "fielder_8": "MLBAM player ID of center fielder",
        "fielder_9": "MLBAM player ID of right fielder",
        "release_pos_y": "Distance from home plate at release point in feet",
        "estimated_ba_using_speedangle": "Expected batting average based on launch speed/angle (xBA)",
        "estimated_woba_using_speedangle": "Expected weighted on-base average based on launch speed/angle (xwOBA)",
        "woba_value": "wOBA value assigned to this outcome",
        "woba_denom": "wOBA denominator (1 if counted, 0 if excluded)",
        "babip_value": "BABIP value Statcast assigns to the plate-appearance outcome (1 = hit on a ball in play other than a home run).",
        "iso_value": "Isolated-power value of the outcome (extra bases: 1 double, 2 triple, 3 home run; 0 otherwise).",
        "launch_speed_angle": "Statcast launch speed/angle category of a batted ball: 1 Weak, 2 Topped, 3 Under, 4 Flare/Burner, 5 Solid Contact, 6 Barrel.",
        "at_bat_number": "At-bat sequence number within the game",
        "pitch_number": "Pitch sequence number within the at-bat",
        "pitch_name": "Full pitch type name (e.g., \"4-Seam Fastball\", \"Slider\")",
        "home_score": "Home team score before this pitch",
        "away_score": "Away team score before this pitch",
        "bat_score": "Batting team score before this pitch",
        "fld_score": "Fielding team score before this pitch",
        "post_away_score": "Away team score after this pitch",
        "post_home_score": "Home team score after this pitch",
        "post_bat_score": "Batting team score after this pitch",
        "post_fld_score": "Fielding team score after this pitch",
        "if_fielding_alignment": "Infield defensive alignment (Standard, Strategic, Shift, etc.)",
        "of_fielding_alignment": "Outfield defensive alignment (Standard, Strategic, Shift, etc.)",
        "spin_axis": "Spin axis orientation in degrees (0-360, with 180=pure backspin, 0=pure topspin)",
        "delta_home_win_exp": "Change in home team win expectancy from this pitch",
        "delta_run_exp": "Change in run expectancy from this pitch",
        "bat_speed": "Bat speed in mph from Statcast bat tracking; empty on pitches without a tracked swing and before 2023 (Statcast bat tracking).",
        "swing_length": "Swing length in feet from Statcast bat tracking; empty on pitches without a tracked swing and before 2023 (Statcast bat tracking).",
        "miss_distance": "Statcast's miss distance on a swinging strike (how far the bat missed the ball); empty on other pitches and before 2023. Savant does not state the unit.",
        "estimated_slg_using_speedangle": "Expected slugging percentage based on launch speed/angle (xSLG)",
        "delta_pitcher_run_exp": "Change in run expectancy attributed to pitcher",
        "hyper_speed": "Statcast's hyper speed: launch_speed with a floor of 88 mph (launch_speed at 88 and above, 88 below). Where launch_speed is empty it is empty or 88 (some foul bunts).",
        "home_score_diff": "Home team score differential (home minus away)",
        "bat_score_diff": "Batting team score differential",
        "home_win_exp": "Home team win expectancy before this pitch (0-1)",
        "bat_win_exp": "Batting team win expectancy before this pitch (0-1)",
        "age_pit_legacy": "Savant's earlier pitcher age field (see age_pit).",
        "age_bat_legacy": "Savant's earlier batter age field (see age_bat).",
        "age_pit": "Pitcher's age as Savant gives it now. age_pit_legacy is Savant's earlier age field; the two can differ by one year.",
        "age_bat": "Batter's age as Savant gives it now. age_bat_legacy is Savant's earlier age field; the two can differ by one year.",
        "n_thruorder_pitcher": "Times pitcher has faced the batting order this game",
        "n_priorpa_thisgame_player_at_bat": "Number of prior plate appearances by this batter in this game",
        "pitcher_days_since_prev_game": "Days since pitcher's previous game appearance",
        "batter_days_since_prev_game": "Days since batter's previous game appearance",
        "pitcher_days_until_next_game": "Days until pitcher's next game appearance",
        "batter_days_until_next_game": "Days until batter's next game appearance",
        "api_break_z_with_gravity": "Vertical break including gravity, in feet; positive = downward.",
        "api_break_x_arm": "Horizontal break in feet toward the pitcher's arm side (positive = arm side).",
        "api_break_x_batter_in": "Horizontal break in feet toward the batter (positive = in on the batter).",
        "arm_angle": "Pitcher's arm angle at release in degrees (Savant); mostly filled from 2020 in this dataset, empty in 2015-2019.",
        "attack_angle": "Bat's vertical attack angle at contact in degrees (Statcast bat tracking); empty on pitches without a tracked swing and before 2023 (Statcast bat tracking).",
        "attack_direction": "Bat's horizontal attack direction in degrees (Statcast bat tracking); empty on pitches without a tracked swing and before 2023 (Statcast bat tracking).",
        "swing_path_tilt": "Tilt of the swing plane in degrees (Statcast bat tracking); empty on pitches without a tracked swing and before 2023 (Statcast bat tracking).",
        "intercept_ball_minus_batter_pos_x_inches": "Statcast bat tracking: horizontal distance in inches between where bat meets ball and the batter's position; empty on pitches without a tracked swing and before 2023 (Statcast bat tracking).",
        "intercept_ball_minus_batter_pos_y_inches": "Statcast bat tracking: depth distance in inches between where bat meets ball and the batter's position; empty on pitches without a tracked swing and before 2023 (Statcast bat tracking)."
    };
    let updated = 0, skipped = 0, failed = 0;
    const headers = document.querySelectorAll('span[title]');
    console.log('Found ' + headers.length + ' columns');

    headers.forEach((header) => {
        const name = header.getAttribute('title');
        if (!columns[name]) { skipped++; return; }
        try {
            const th = header.closest('th');
            const input = th.querySelector('input[placeholder="Please enter a description"]');
            const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            setter.call(input, columns[name]);
            input.dispatchEvent(new Event('input', {bubbles: true}));
            input.dispatchEvent(new Event('change', {bubbles: true}));
            input.dispatchEvent(new Event('blur', {bubbles: true}));
            console.log('[OK] ' + name);
            updated++;
        } catch(e) {
            console.error('[ERROR] ' + name + ': ' + e.message);
            failed++;
        }
    });

    console.log('\n=== SUMMARY ===');
    console.log('Updated: ' + updated);
    console.log('Skipped: ' + skipped);
    console.log('Failed: ' + failed);
    console.log('\n[IMPORTANT] Please review and click SAVE!');
})();
