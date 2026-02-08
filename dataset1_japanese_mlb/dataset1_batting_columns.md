# Dataset 1: Japanese MLB Batters - Column Descriptions

## File: japanese_mlb_batting.csv

---

### **pitch_type**
```
Pitch type abbreviation (FF=Four-Seam Fastball, SL=Slider, CH=Changeup, CU=Curveball, etc.) - Classified by Statcast's pitch tracking system
```

### **game_date**
```
Date of the game (YYYY-MM-DD format)
```

### **release_speed**
```
Pitch velocity at release point in miles per hour (mph)
```

### **release_pos_x**
```
Horizontal release point in feet (catcher's perspective: negative=left, positive=right)
```

### **release_pos_z**
```
Vertical release point in feet (height above ground)
```

### **player_name**
```
Batter's full name
```

### **batter**
```
MLBAM player ID of the batter (matches 'mlbam_id' in players.csv)
```

### **pitcher**
```
MLBAM player ID of the pitcher throwing this pitch
```

### **events**
```
At-bat outcome (e.g., single, strikeout, home_run, walk) - Only on final pitch of plate appearance
```

### **description**
```
Individual pitch result (ball, called_strike, swinging_strike, hit_into_play, foul, etc.)
```

### **spin_dir**
```
Deprecated spin direction field - Use spin_axis instead
```

### **spin_rate_deprecated**
```
Deprecated spin rate field - Use release_spin_rate instead
```

### **break_angle_deprecated**
```
Deprecated break angle field - Use api_break fields instead
```

### **break_length_deprecated**
```
Deprecated break length field - Use api_break fields instead
```

### **zone**
```
Strike zone location (1-9=strike zone grid, 11-14=outside zone)
```

### **des**
```
Detailed text description of the play
```

### **game_type**
```
Type of game (R=Regular season, P=Playoffs, S=Spring training, etc.)
```

### **stand**
```
Batter's stance (R=right-handed, L=left-handed)
```

### **p_throws**
```
Pitcher's throwing hand (R=right, L=left)
```

### **home_team**
```
Home team abbreviation (e.g., NYY, LAD)
```

### **away_team**
```
Away team abbreviation
```

### **type**
```
Pitch outcome category (S=strike, B=ball, X=in play)
```

### **hit_location**
```
Fielder position number where ball was hit (1=pitcher, 2=catcher, 3=1B, etc.)
```

### **bb_type**
```
Batted ball type (ground_ball, line_drive, fly_ball, popup)
```

### **balls**
```
Number of balls in the count (0-3)
```

### **strikes**
```
Number of strikes in the count (0-2)
```

### **game_year**
```
Season year of the game
```

### **pfx_x**
```
Horizontal movement in inches (catcher's perspective: negative=left, positive=right) due to spin
```

### **pfx_z**
```
Vertical movement in inches (positive=rise, negative=drop) due to spin, gravity-adjusted
```

### **plate_x**
```
Horizontal position at home plate in feet (catcher's perspective: negative=left, positive=right)
```

### **plate_z**
```
Vertical position at home plate in feet (height above ground)
```

### **on_3b**
```
MLBAM player ID of runner on 3rd base (NaN if empty)
```

### **on_2b**
```
MLBAM player ID of runner on 2nd base (NaN if empty)
```

### **on_1b**
```
MLBAM player ID of runner on 1st base (NaN if empty)
```

### **outs_when_up**
```
Number of outs when batter came to plate (0-2)
```

### **inning**
```
Inning number
```

### **inning_topbot**
```
Top or Bottom of inning
```

### **hc_x**
```
Hit coordinate X position (pixel coordinates on field diagram)
```

### **hc_y**
```
Hit coordinate Y position (pixel coordinates on field diagram)
```

### **tfs_deprecated**
```
Deprecated timestamp field
```

### **tfs_zulu_deprecated**
```
Deprecated Zulu timestamp field
```

### **umpire**
```
Home plate umpire's MLBAM ID
```

### **sv_id**
```
Savant ID for video replay
```

### **vx0**
```
Initial velocity in X direction (feet per second) at y=50 feet
```

### **vy0**
```
Initial velocity in Y direction (feet per second) at y=50 feet
```

### **vz0**
```
Initial velocity in Z direction (feet per second) at y=50 feet
```

### **ax**
```
Acceleration in X direction (feet per second squared)
```

### **ay**
```
Acceleration in Y direction (feet per second squared)
```

### **az**
```
Acceleration in Z direction (feet per second squared)
```

### **sz_top**
```
Top of batter's strike zone in feet (varies by batter height)
```

### **sz_bot**
```
Bottom of batter's strike zone in feet (varies by batter height)
```

### **hit_distance_sc**
```
Projected hit distance in feet (Statcast calculation)
```

### **launch_speed**
```
Exit velocity off the bat in miles per hour (mph)
```

### **launch_angle**
```
Vertical launch angle in degrees (negative=ground ball, positive=fly ball)
```

### **effective_speed**
```
Perceived velocity adjusted for release point extension (mph)
```

### **release_spin_rate**
```
Spin rate at release in revolutions per minute (RPM)
```

### **release_extension**
```
Release point extension toward home plate in feet (distance from rubber)
```

### **game_pk**
```
Unique game identifier in MLB's database
```

### **fielder_2**
```
MLBAM player ID of catcher
```

### **fielder_3**
```
MLBAM player ID of first baseman
```

### **fielder_4**
```
MLBAM player ID of second baseman
```

### **fielder_5**
```
MLBAM player ID of third baseman
```

### **fielder_6**
```
MLBAM player ID of shortstop
```

### **fielder_7**
```
MLBAM player ID of left fielder
```

### **fielder_8**
```
MLBAM player ID of center fielder
```

### **fielder_9**
```
MLBAM player ID of right fielder
```

### **release_pos_y**
```
Distance from home plate at release point in feet
```

### **estimated_ba_using_speedangle**
```
Expected batting average based on launch speed/angle (xBA)
```

### **estimated_woba_using_speedangle**
```
Expected weighted on-base average based on launch speed/angle (xwOBA)
```

### **woba_value**
```
wOBA value assigned to this outcome
```

### **woba_denom**
```
wOBA denominator (1 if counted, 0 if excluded)
```

### **babip_value**
```
BABIP value assigned (1 if ball in play, 0 otherwise)
```

### **iso_value**
```
Isolated power value (total bases minus hits)
```

### **launch_speed_angle**
```
Launch speed/angle classification (1-6 scale for barrels, weak contact, etc.)
```

### **at_bat_number**
```
At-bat sequence number within the game
```

### **pitch_number**
```
Pitch sequence number within the at-bat
```

### **pitch_name**
```
Full pitch type name (e.g., "4-Seam Fastball", "Slider")
```

### **home_score**
```
Home team score before this pitch
```

### **away_score**
```
Away team score before this pitch
```

### **bat_score**
```
Batting team score before this pitch
```

### **fld_score**
```
Fielding team score before this pitch
```

### **post_away_score**
```
Away team score after this pitch
```

### **post_home_score**
```
Home team score after this pitch
```

### **post_bat_score**
```
Batting team score after this pitch
```

### **post_fld_score**
```
Fielding team score after this pitch
```

### **if_fielding_alignment**
```
Infield defensive alignment (Standard, Strategic, Shift, etc.)
```

### **of_fielding_alignment**
```
Outfield defensive alignment (Standard, Strategic, Shift, etc.)
```

### **spin_axis**
```
Spin axis orientation in degrees (0-360, with 180=pure backspin, 0=pure topspin)
```

### **delta_home_win_exp**
```
Change in home team win expectancy from this pitch
```

### **delta_run_exp**
```
Change in run expectancy from this pitch
```

### **bat_speed**
```
Bat speed at contact in miles per hour (mph) - Available from 2024 onward
```

### **swing_length**
```
Length of bat's path to contact in feet - Available from 2024 onward
```

### **estimated_slg_using_speedangle**
```
Expected slugging percentage based on launch speed/angle (xSLG)
```

### **delta_pitcher_run_exp**
```
Change in run expectancy attributed to pitcher
```

### **hyper_speed**
```
Combined metric of bat speed and swing efficiency
```

### **home_score_diff**
```
Home team score differential (home minus away)
```

### **bat_score_diff**
```
Batting team score differential
```

### **home_win_exp**
```
Home team win expectancy before this pitch (0-1)
```

### **bat_win_exp**
```
Batting team win expectancy before this pitch (0-1)
```

### **age_pit_legacy**
```
Legacy pitcher age calculation (deprecated)
```

### **age_bat_legacy**
```
Legacy batter age calculation (deprecated)
```

### **age_pit**
```
Pitcher's age at time of game
```

### **age_bat**
```
Batter's age at time of game
```

### **n_thruorder_pitcher**
```
Times pitcher has faced the batting order this game
```

### **n_priorpa_thisgame_player_at_bat**
```
Number of prior plate appearances by this batter in this game
```

### **pitcher_days_since_prev_game**
```
Days since pitcher's previous game appearance
```

### **batter_days_since_prev_game**
```
Days since batter's previous game appearance
```

### **pitcher_days_until_next_game**
```
Days until pitcher's next game appearance
```

### **batter_days_until_next_game**
```
Days until batter's next game appearance
```

### **api_break_z_with_gravity**
```
Vertical break including gravity effect in inches
```

### **api_break_x_arm**
```
Horizontal break from pitcher's arm side perspective in inches
```

### **api_break_x_batter_in**
```
Horizontal break toward/away from batter in inches (positive=toward batter's hands)
```

### **arm_angle**
```
Pitcher's arm angle at release in degrees
```

### **attack_angle**
```
Batter's attack angle at contact in degrees
```

### **attack_direction**
```
Direction of bat's attack path in degrees
```

### **swing_path_tilt**
```
Tilt of the swing plane in degrees
```

### **intercept_ball_minus_batter_pos_x_inches**
```
Horizontal distance between ball intercept and batter position in inches
```

### **intercept_ball_minus_batter_pos_y_inches**
```
Depth distance between ball intercept and batter position in inches
```

### **player_name_eng**
```
Batter's name in English transliteration
```

---

## Usage Note
Copy and paste each description into the corresponding column in Kaggle Dataset settings.
