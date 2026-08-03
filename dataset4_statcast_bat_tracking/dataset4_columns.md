# Dataset 4: MLB Statcast + Bat Tracking (2024-2025) - Column Descriptions

### **bat_speed**
```
Bat speed at contact in miles per hour (mph). Measured by high-speed cameras. Available for approximately 46% of pitches. Higher values indicate faster swing velocity.
```

### **swing_length**
```
Length of the bat's path to contact in feet. Shorter swing length generally indicates more efficient bat path and better swing mechanics.
```

### **swing_path_tilt**
```
Angle of the swing plane in degrees. Measured relative to horizontal plane. Positive values indicate uppercut swing, negative values indicate downward swing.
```

### **attack_angle**
```
Angle of bat at contact relative to pitch trajectory in degrees. Measures how well the swing plane matches the pitch plane.
```

### **attack_direction**
```
Horizontal angle of bat at contact in degrees. Indicates pull vs opposite field swing direction.
```

### **intercept_ball_minus_batter_pos_x_inches**
```
Horizontal distance in inches between ball intercept point and batter position at contact.
```

### **intercept_ball_minus_batter_pos_y_inches**
```
Vertical distance in inches between ball intercept point and batter position at contact.
```

### **pitch_type**
```
Statcast pitch classification code. FF=Four-seam fastball, SL=Slider, CH=Changeup, CU=Curveball, SI=Sinker, FC=Cutter, FS=Splitter, KC=Knuckle Curve, EP=Eephus, FO=Forkball, KN=Knuckleball, ST=Sweeper.
```

### **pitch_name**
```
Human-readable pitch type name (e.g. "4-Seam Fastball", "Slider", "Changeup").
```

### **game_date**
```
Date of the game in YYYY-MM-DD format.
```

### **game_year**
```
Season year (2024 or 2025).
```

### **game_type**
```
Type of game. R=Regular season, F=Wild card, D=Division series, L=League championship, W=World Series, S=Spring training, E=Exhibition.
```

### **game_pk**
```
Unique MLB Advanced Media game identifier.
```

### **inning**
```
Inning number (1-9 for regulation, 10+ for extra innings).
```

### **inning_topbot**
```
Top or Bot (bottom) of inning.
```

### **at_bat_number**
```
Sequential at-bat number within the game.
```

### **pitch_number**
```
Sequential pitch number within the at-bat.
```

### **player_name**
```
Name of the batter in Last, First format.
```

### **batter**
```
MLB Advanced Media player ID for the batter.
```

### **pitcher**
```
MLB Advanced Media player ID for the pitcher.
```

### **stand**
```
Batter's stance. R=Right-handed, L=Left-handed.
```

### **p_throws**
```
Pitcher's throwing hand. R=Right, L=Left.
```

### **age_bat**
```
Batter's age in years (decimal, e.g. 27.5).
```

### **age_pit**
```
Pitcher's age in years (decimal).
```

### **age_bat_legacy**
```
Batter's age in legacy integer format.
```

### **age_pit_legacy**
```
Pitcher's age in legacy integer format.
```

### **home_team**
```
Home team abbreviation (e.g. NYY, LAD, BOS).
```

### **away_team**
```
Away team abbreviation.
```

### **home_score**
```
Home team score at time of pitch.
```

### **away_score**
```
Away team score at time of pitch.
```

### **bat_score**
```
Batting team score at time of pitch.
```

### **fld_score**
```
Fielding team score at time of pitch.
```

### **post_home_score**
```
Home team score after the play.
```

### **post_away_score**
```
Away team score after the play.
```

### **post_bat_score**
```
Batting team score after the play.
```

### **post_fld_score**
```
Fielding team score after the play.
```

### **home_score_diff**
```
Home team run differential (home score minus away score).
```

### **bat_score_diff**
```
Batting team run differential.
```

### **release_speed**
```
Pitch velocity at release point in miles per hour (mph). Measured 50 feet from home plate.
```

### **release_pos_x**
```
Horizontal release point in feet from catcher's perspective. Negative = pitcher's left, positive = pitcher's right.
```

### **release_pos_y**
```
Distance from home plate to release point in feet. Typically 54-56 feet.
```

### **release_pos_z**
```
Vertical release point in feet above ground. Typically 5-7 feet for overhand, 4-5 feet for sidearm.
```

### **release_extension**
```
Distance in feet from rubber to release point. Longer extension means ball is released closer to home plate.
```

### **effective_speed**
```
Perceived velocity accounting for release extension. Calculated as: release_speed * 55 / (55 + release_extension).
```

### **pfx_x**
```
Horizontal movement in inches with gravity removed. Measured at home plate. Negative = moves toward RHB, positive = moves toward LHB.
```

### **pfx_z**
```
Vertical movement in inches with gravity removed. Positive = rises more than gravity, negative = drops more.
```

### **plate_x**
```
Horizontal location as ball crosses plate in feet from catcher's perspective. 0 = center, negative = inside to RHB, positive = outside to RHB.
```

### **plate_z**
```
Vertical location as ball crosses plate in feet above ground. Strike zone typically 1.5-3.5 feet.
```

### **vx0**
```
Horizontal velocity component at release in feet per second.
```

### **vy0**
```
Forward velocity component at release in feet per second (toward home plate).
```

### **vz0**
```
Vertical velocity component at release in feet per second.
```

### **ax**
```
Horizontal acceleration in feet per second squared.
```

### **ay**
```
Forward acceleration in feet per second squared.
```

### **az**
```
Vertical acceleration in feet per second squared.
```

### **release_spin_rate**
```
Spin rate at release in revolutions per minute (rpm). Higher spin typically creates more movement.
```

### **spin_axis**
```
Tilt of spin axis in degrees (0-360). 180° = pure backspin (fastball), 0° = pure topspin (12-6 curveball).
```

### **spin_dir**
```
Direction of spin (deprecated field).
```

### **spin_rate_deprecated**
```
Old spin rate field (deprecated, use release_spin_rate instead).
```

### **arm_angle**
```
Angle of pitcher's arm slot in degrees. 90° = overhand, 45° = three-quarters, 0° = sidearm.
```

### **break_angle_deprecated**
```
Old break angle metric (deprecated).
```

### **break_length_deprecated**
```
Old break length metric (deprecated).
```

### **zone**
```
Strike zone location. 1-9 = in zone (1=top-left to 9=bottom-right), 11=above zone, 12=below zone, 13=inside, 14=outside.
```

### **sz_top**
```
Top of batter's strike zone in feet above ground. Varies by batter height and stance.
```

### **sz_bot**
```
Bottom of batter's strike zone in feet above ground.
```

### **type**
```
Pitch outcome type. S=Strike, B=Ball, X=In play.
```

### **description**
```
Detailed pitch outcome (e.g. "called_strike", "swinging_strike", "ball", "foul", "hit_into_play").
```

### **events**
```
Outcome of plate appearance if pitch ended the at-bat (single, double, triple, home_run, strikeout, walk, field_out, etc.). Null for pitches that did not end PA.
```

### **des**
```
Natural language description of the play.
```

### **launch_speed**
```
Exit velocity of batted ball in miles per hour (mph). Higher values correlate with harder hits and better outcomes.
```

### **launch_angle**
```
Vertical angle of batted ball in degrees. Optimal for home runs is 25-35°. Line drives: 10-25°, fly balls: 25-50°, ground balls: <10°.
```

### **launch_speed_angle**
```
Launch speed/angle classification. 1=Weak, 2=Topped, 3=Under, 4=Flare/Burner, 5=Solid Contact, 6=Barrel (26-30° with 98+ mph exit velo).
```

### **hit_distance_sc**
```
Projected hit distance in feet based on launch speed and angle using physics model.
```

### **hc_x**
```
Horizontal coordinate of batted ball location in pixels for spray chart visualization.
```

### **hc_y**
```
Vertical coordinate of batted ball location in pixels for spray chart visualization.
```

### **bb_type**
```
Batted ball type: ground_ball, line_drive, fly_ball, or popup.
```

### **hit_location**
```
Fielder position number where ball was hit. 1=Pitcher, 2=Catcher, 3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF.
```

### **estimated_ba_using_speedangle**
```
Expected batting average based on launch speed and angle from historical outcomes. Scale: 0-1.0 (e.g. 0.500 = 50% hit probability).
```

### **estimated_woba_using_speedangle**
```
Expected weighted on-base average (wOBA) based on launch speed and angle. Higher = better expected outcomes. Scale: 0-2.0 (league avg ~0.320).
```

### **estimated_slg_using_speedangle**
```
Expected slugging percentage based on launch speed and angle. Scale: 0-4.0 (1.0=single, 2.0=double, 3.0=triple, 4.0=HR).
```

### **woba_value**
```
Actual wOBA value for the outcome weighted by run expectancy.
```

### **woba_denom**
```
Denominator for wOBA calculation. 1 for most PAs, 0 for sac bunts/flies.
```

### **babip_value**
```
Batting average on balls in play value. 1 for hit, 0 for out. Excludes home runs and strikeouts.
```

### **iso_value**
```
Isolated power value. Extra bases on hit: 0=out, 1=single, 2=double, 3=triple, 4=home run minus 1.
```

### **home_win_exp**
```
Home team win expectancy (0-1) at time of pitch based on score, inning, outs, and runners.
```

### **bat_win_exp**
```
Batting team win expectancy (0-1).
```

### **delta_home_win_exp**
```
Change in home team win expectancy after the play. Positive = increased win probability, negative = decreased.
```

### **delta_run_exp**
```
Change in run expectancy after the play measured in expected runs scored in remainder of inning.
```

### **delta_pitcher_run_exp**
```
Change in run expectancy from pitcher's perspective (inverse of delta_run_exp).
```

### **balls**
```
Number of balls in the count (0-3).
```

### **strikes**
```
Number of strikes in the count (0-2).
```

### **outs_when_up**
```
Number of outs when batter came to plate (0-2).
```

### **on_1b**
```
Player ID of runner on first base. Null if empty.
```

### **on_2b**
```
Player ID of runner on second base. Null if empty.
```

### **on_3b**
```
Player ID of runner on third base. Null if empty.
```

### **fielder_2**
```
Player ID of catcher.
```

### **fielder_3**
```
Player ID of first baseman.
```

### **fielder_4**
```
Player ID of second baseman.
```

### **fielder_5**
```
Player ID of third baseman.
```

### **fielder_6**
```
Player ID of shortstop.
```

### **fielder_7**
```
Player ID of left fielder.
```

### **fielder_8**
```
Player ID of center fielder.
```

### **fielder_9**
```
Player ID of right fielder.
```

### **if_fielding_alignment**
```
Infield positioning: Standard, Infield shift, or Strategic.
```

### **of_fielding_alignment**
```
Outfield positioning: Standard, 4th outfielder, or Strategic.
```

### **sv_id**
```
Statcast tracking system pitch identifier.
```

### **umpire**
```
Home plate umpire MLB ID.
```

### **tfs_deprecated**
```
Deprecated timestamp field.
```

### **tfs_zulu_deprecated**
```
Deprecated Zulu time field.
```

### **n_thruorder_pitcher**
```
Times pitcher has faced batting order (1st, 2nd, 3rd+ time through). Performance often declines on 3rd+ time.
```

### **n_priorpa_thisgame_player_at_bat**
```
Number of prior plate appearances by batter in current game (0-based: 0=first PA, 1=second PA, etc.).
```

### **pitcher_days_since_prev_game**
```
Days since pitcher's previous appearance for rest/fatigue analysis.
```

### **batter_days_since_prev_game**
```
Days since batter's previous game.
```

### **pitcher_days_until_next_game**
```
Days until pitcher's next appearance.
```

### **batter_days_until_next_game**
```
Days until batter's next game.
```

### **api_break_z_with_gravity**
```
Vertical break with gravity in inches. How much ball drops compared to straight line with gravity.
```

### **api_break_x_arm**
```
Horizontal break from pitcher's arm side perspective in inches.
```

### **api_break_x_batter_in**
```
Horizontal break from batter's perspective in inches (arm-side break).
```

### **hyper_speed**
```
Hyperspace speed metric (experimental). Combines velocity components in multi-dimensional space.
```
