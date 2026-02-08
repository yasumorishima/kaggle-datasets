# Dataset 3: MLB Pitcher Arsenal Evolution (2020-2025) - Column Descriptions

## File: pitcher_arsenal_evolution_2020_2025.csv

---

### **player_id**
```
Unique MLB player ID (integer). Used to join with other Statcast datasets.
```

### **player_name**
```
Player's full name (First Last format).
```

### **season**
```
MLB season year (2020-2025).
```

### **FF_usage_pct**
```
Four-Seam Fastball usage percentage. Represents how often the pitcher throws this pitch type (0-100%).
```

### **FF_avg_speed**
```
Four-Seam Fastball average velocity in mph. Typically 90-100 mph for MLB pitchers.
```

### **FF_whiff_rate**
```
Four-Seam Fastball swing-and-miss rate. Calculated as (swings and misses) / (total swings).
```

### **FF_avg_pfx_x**
```
Four-Seam Fastball average horizontal movement in inches. Positive values indicate movement toward the pitcher's arm side, negative values toward glove side.
```

### **FF_avg_pfx_z**
```
Four-Seam Fastball average vertical movement in inches (gravity-adjusted). Positive values indicate "rise," negative values indicate drop.
```

### **SI_usage_pct**
```
Sinker usage percentage. Sinkers typically have more horizontal movement and less vertical movement than four-seam fastballs.
```

### **SI_avg_speed**
```
Sinker average velocity in mph. Usually 89-96 mph.
```

### **FC_usage_pct**
```
Cutter usage percentage. Cutters move toward the pitcher's glove side with late break.
```

### **FC_avg_speed**
```
Cutter average velocity in mph. Typically 85-92 mph.
```

### **SL_usage_pct**
```
Slider usage percentage. One of the most effective swing-and-miss pitches in modern MLB.
```

### **SL_avg_speed**
```
Slider average velocity in mph. Usually 80-88 mph.
```

### **SL_whiff_rate**
```
Slider swing-and-miss rate. Often the highest whiff rate among pitch types.
```

### **SL_avg_pfx_x**
```
Slider average horizontal movement in inches. Typically large negative values (glove-side break).
```

### **SL_avg_pfx_z**
```
Slider average vertical movement in inches. Usually negative (downward movement).
```

### **CU_usage_pct**
```
Curveball usage percentage. Traditional breaking ball with vertical drop.
```

### **CU_avg_speed**
```
Curveball average velocity in mph. Typically 70-80 mph.
```

### **CU_whiff_rate**
```
Curveball swing-and-miss rate. Effective due to significant drop and slower speed.
```

### **CH_usage_pct**
```
Changeup usage percentage. Off-speed pitch designed to look like a fastball.
```

### **CH_avg_speed**
```
Changeup average velocity in mph. Usually 78-86 mph (6-10 mph slower than fastball).
```

### **CH_whiff_rate**
```
Changeup swing-and-miss rate. Effective against opposite-handed batters.
```

### **FS_usage_pct**
```
Splitter usage percentage. "Forkball-type" pitch with sharp downward movement.
```

### **FS_avg_speed**
```
Splitter average velocity in mph. Typically 82-88 mph.
```

### **FS_whiff_rate**
```
Splitter swing-and-miss rate. Often the highest whiff rate due to late tumbling action.
```

### **KC_usage_pct**
```
Knuckle Curve usage percentage. Curveball variant with knuckleball grip.
```

### **FO_usage_pct**
```
Forkball usage percentage. Rare pitch with extreme tumbling action (used by Senga, Darvish).
```

### **UN_usage_pct**
```
Unclassified pitch usage percentage. Pitches that don't fit standard classifications.
```
