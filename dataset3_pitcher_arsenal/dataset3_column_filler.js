// Kaggle Column Description Auto-Filler
(function() {
    const columns = {
  "player_id": "Unique MLB player ID (integer). Used to join with other Statcast datasets.",
  "player_name": "Player's full name (First Last format).",
  "season": "MLB season year (2020-2025).",
  "CH_usage_pct": "Changeup usage percentage (0-100%). Off-speed pitch designed to look like a fastball with late downward movement.",
  "CS_usage_pct": "Slow Curve usage percentage (0-100%). Variation of curveball with slower speed and more looping trajectory.",
  "CU_usage_pct": "Curveball usage percentage (0-100%). Traditional breaking ball with vertical drop and 12-to-6 movement.",
  "EP_usage_pct": "Eephus usage percentage (0-100%). Extremely slow pitch with high arc, rarely used (novelty pitch).",
  "FA_usage_pct": "Fastball (generic) usage percentage (0-100%). Used when Statcast cannot distinguish between FF/SI.",
  "FC_usage_pct": "Cutter usage percentage (0-100%). Fastball variant that moves toward pitcher's glove side with late break.",
  "FF_usage_pct": "Four-Seam Fastball usage percentage (0-100%). Most common pitch type with straight trajectory and high velocity.",
  "FO_usage_pct": "Forkball usage percentage (0-100%). Rare pitch with extreme tumbling action and sharp downward movement.",
  "FS_usage_pct": "Splitter usage percentage (0-100%). Fastball-speed pitch with late downward tumble, similar to forkball.",
  "KC_usage_pct": "Knuckle Curve usage percentage (0-100%). Curveball variant thrown with knuckleball grip for added movement.",
  "KN_usage_pct": "Knuckleball usage percentage (0-100%). Extremely rare pitch with unpredictable movement and minimal spin.",
  "PO_usage_pct": "Pitch Out usage percentage (0-100%). Intentionally thrown outside strike zone to catch baserunners stealing.",
  "SC_usage_pct": "Screwball usage percentage (0-100%). Extremely rare pitch with movement opposite to slider/curveball.",
  "SI_usage_pct": "Sinker usage percentage (0-100%). Two-seam fastball with downward and armside horizontal movement.",
  "SL_usage_pct": "Slider usage percentage (0-100%). Breaking ball with lateral movement and downward tilt, modern MLB's most popular pitch.",
  "ST_usage_pct": "Sweeper usage percentage (0-100%). Slider variant with more horizontal movement and less vertical drop (modern classification).",
  "SV_usage_pct": "Slurve usage percentage (0-100%). Hybrid pitch between slider and curveball with both lateral and vertical movement.",
  "UN_usage_pct": "Unclassified pitch usage percentage (0-100%). Pitches that don't fit standard classifications or have ambiguous characteristics.",
  "CH_avg_speed": "Changeup average velocity in mph. Typically 78-86 mph (6-10 mph slower than fastball).",
  "CS_avg_speed": "Slow Curve average velocity in mph. Usually 65-72 mph.",
  "CU_avg_speed": "Curveball average velocity in mph. Typically 70-80 mph.",
  "EP_avg_speed": "Eephus average velocity in mph. Extremely slow, often 50-65 mph.",
  "FA_avg_speed": "Fastball (generic) average velocity in mph. Typically 90-95 mph.",
  "FC_avg_speed": "Cutter average velocity in mph. Typically 85-92 mph.",
  "FF_avg_speed": "Four-Seam Fastball average velocity in mph. Typically 90-100 mph for MLB pitchers.",
  "FO_avg_speed": "Forkball average velocity in mph. Usually 80-86 mph.",
  "FS_avg_speed": "Splitter average velocity in mph. Typically 82-88 mph.",
  "KC_avg_speed": "Knuckle Curve average velocity in mph. Usually 72-78 mph.",
  "KN_avg_speed": "Knuckleball average velocity in mph. Typically 60-75 mph.",
  "PO_avg_speed": "Pitch Out average velocity in mph. Variable, often 75-85 mph.",
  "SC_avg_speed": "Screwball average velocity in mph. Typically 70-80 mph.",
  "SI_avg_speed": "Sinker average velocity in mph. Usually 89-96 mph.",
  "SL_avg_speed": "Slider average velocity in mph. Usually 80-88 mph.",
  "ST_avg_speed": "Sweeper average velocity in mph. Typically 78-84 mph.",
  "SV_avg_speed": "Slurve average velocity in mph. Usually 75-82 mph.",
  "UN_avg_speed": "Unclassified pitch average velocity in mph. Variable depending on pitch type.",
  "CH_avg_spin": "Changeup average spin rate in rpm. Typically 1400-1900 rpm, lower than fastballs to create speed differential effect.",
  "CS_avg_spin": "Slow Curve average spin rate in rpm. Usually 1800-2400 rpm.",
  "CU_avg_spin": "Curveball average spin rate in rpm. Typically 2400-3000 rpm, high spin creates sharp downward break.",
  "EP_avg_spin": "Eephus average spin rate in rpm. Very low spin, typically 500-1200 rpm.",
  "FA_avg_spin": "Fastball (generic) average spin rate in rpm. Typically 2000-2500 rpm.",
  "FC_avg_spin": "Cutter average spin rate in rpm. Usually 2200-2700 rpm.",
  "FF_avg_spin": "Four-Seam Fastball average spin rate in rpm. Typically 2100-2600 rpm, high spin creates \"rising\" effect.",
  "FO_avg_spin": "Forkball average spin rate in rpm. Very low spin, typically 800-1400 rpm, creates tumbling action.",
  "FS_avg_spin": "Splitter average spin rate in rpm. Low spin, typically 1200-1800 rpm, creates late downward movement.",
  "KC_avg_spin": "Knuckle Curve average spin rate in rpm. Usually 2000-2600 rpm.",
  "KN_avg_spin": "Knuckleball average spin rate in rpm. Extremely low spin, typically 50-300 rpm, creates unpredictable movement.",
  "PO_avg_spin": "Pitch Out average spin rate in rpm. Variable, typically 1800-2300 rpm.",
  "SC_avg_spin": "Screwball average spin rate in rpm. Usually 1800-2400 rpm.",
  "SI_avg_spin": "Sinker average spin rate in rpm. Typically 1900-2400 rpm, lower than four-seam to create sink.",
  "SL_avg_spin": "Slider average spin rate in rpm. Usually 2200-2800 rpm, spin axis creates lateral break.",
  "ST_avg_spin": "Sweeper average spin rate in rpm. Typically 2100-2600 rpm.",
  "SV_avg_spin": "Slurve average spin rate in rpm. Usually 2000-2600 rpm.",
  "UN_avg_spin": "Unclassified pitch average spin rate in rpm. Variable depending on pitch type.",
  "CH_whiff_rate": "Changeup swing-and-miss rate (0-1). Calculated as (swings and misses) / (total swings). Effective against opposite-handed batters.",
  "CS_whiff_rate": "Slow Curve swing-and-miss rate (0-1). Higher values indicate better deception due to speed differential and movement.",
  "CU_whiff_rate": "Curveball swing-and-miss rate (0-1). Effective due to significant vertical drop and slower speed compared to fastballs.",
  "EP_whiff_rate": "Eephus swing-and-miss rate (0-1). Variable effectiveness, primarily used for surprise factor.",
  "FA_whiff_rate": "Fastball (generic) swing-and-miss rate (0-1). Generally lower than breaking balls.",
  "FC_whiff_rate": "Cutter swing-and-miss rate (0-1). Effective due to late lateral movement.",
  "FF_whiff_rate": "Four-Seam Fastball swing-and-miss rate (0-1). Lower than breaking balls but effective when located well.",
  "FO_whiff_rate": "Forkball swing-and-miss rate (0-1). Extremely high whiff rate due to tumbling action and speed differential.",
  "FS_whiff_rate": "Splitter swing-and-miss rate (0-1). Often the highest whiff rate among pitch types due to late tumbling action.",
  "KC_whiff_rate": "Knuckle Curve swing-and-miss rate (0-1). Effective due to added movement from knuckleball grip.",
  "KN_whiff_rate": "Knuckleball swing-and-miss rate (0-1). Variable due to unpredictable movement pattern.",
  "PO_whiff_rate": "Pitch Out swing-and-miss rate (0-1). Not applicable to normal pitching analysis (defensive play).",
  "SC_whiff_rate": "Screwball swing-and-miss rate (0-1). Rare pitch, effectiveness varies.",
  "SI_whiff_rate": "Sinker swing-and-miss rate (0-1). Generally lower than breaking balls, primarily induces ground balls.",
  "SL_whiff_rate": "Slider swing-and-miss rate (0-1). Often the highest whiff rate among commonly used pitches due to lateral movement.",
  "ST_whiff_rate": "Sweeper swing-and-miss rate (0-1). High whiff rate due to extreme horizontal movement.",
  "SV_whiff_rate": "Slurve swing-and-miss rate (0-1). Moderate to high whiff rate combining slider and curve characteristics.",
  "UN_whiff_rate": "Unclassified pitch swing-and-miss rate (0-1). Variable depending on pitch characteristics.",
  "CH_avg_pfx_x": "Changeup average horizontal movement in inches. Positive values indicate movement toward pitcher's arm side, negative toward glove side.",
  "CS_avg_pfx_x": "Slow Curve average horizontal movement in inches. Typically moderate glove-side movement.",
  "CU_avg_pfx_x": "Curveball average horizontal movement in inches. Usually moderate glove-side movement combined with vertical drop.",
  "EP_avg_pfx_x": "Eephus average horizontal movement in inches. Variable and unpredictable.",
  "FA_avg_pfx_x": "Fastball (generic) average horizontal movement in inches. Variable depending on specific fastball type.",
  "FC_avg_pfx_x": "Cutter average horizontal movement in inches. Typically 2-6 inches glove-side (negative values).",
  "FF_avg_pfx_x": "Four-Seam Fastball average horizontal movement in inches. Typically slight arm-side movement for right-handed pitchers.",
  "FO_avg_pfx_x": "Forkball average horizontal movement in inches. Minimal horizontal movement, primarily vertical drop.",
  "FS_avg_pfx_x": "Splitter average horizontal movement in inches. Typically slight arm-side movement.",
  "KC_avg_pfx_x": "Knuckle Curve average horizontal movement in inches. Moderate glove-side movement.",
  "KN_avg_pfx_x": "Knuckleball average horizontal movement in inches. Highly variable and unpredictable.",
  "PO_avg_pfx_x": "Pitch Out average horizontal movement in inches. Variable, not relevant for performance analysis.",
  "SC_avg_pfx_x": "Screwball average horizontal movement in inches. Opposite direction to typical breaking balls.",
  "SI_avg_pfx_x": "Sinker average horizontal movement in inches. Typically 6-12 inches arm-side (positive for RHP).",
  "SL_avg_pfx_x": "Slider average horizontal movement in inches. Typically large glove-side movement (negative values, 4-10 inches).",
  "ST_avg_pfx_x": "Sweeper average horizontal movement in inches. Extreme glove-side movement (8-16 inches).",
  "SV_avg_pfx_x": "Slurve average horizontal movement in inches. Moderate glove-side movement between slider and curve.",
  "UN_avg_pfx_x": "Unclassified pitch average horizontal movement in inches. Variable depending on pitch type.",
  "CH_avg_pfx_z": "Changeup average vertical movement in inches (gravity-adjusted). Typically negative values (downward drop).",
  "CS_avg_pfx_z": "Slow Curve average vertical movement in inches. Large negative values (significant downward drop).",
  "CU_avg_pfx_z": "Curveball average vertical movement in inches. Large negative values (12-to-6 drop, typically -5 to -12 inches).",
  "EP_avg_pfx_z": "Eephus average vertical movement in inches. Extreme downward drop due to gravity arc.",
  "FA_avg_pfx_z": "Fastball (generic) average vertical movement in inches. Variable depending on specific fastball type.",
  "FC_avg_pfx_z": "Cutter average vertical movement in inches. Typically slight positive values (less \"rise\" than four-seam).",
  "FF_avg_pfx_z": "Four-Seam Fastball average vertical movement in inches. Positive values indicate \"rise\" effect (typically +8 to +14 inches).",
  "FO_avg_pfx_z": "Forkball average vertical movement in inches. Extreme negative values (sharp downward drop).",
  "FS_avg_pfx_z": "Splitter average vertical movement in inches. Negative values (sharp downward tumble, typically -3 to -8 inches).",
  "KC_avg_pfx_z": "Knuckle Curve average vertical movement in inches. Large negative values (significant drop).",
  "KN_avg_pfx_z": "Knuckleball average vertical movement in inches. Highly variable and unpredictable.",
  "PO_avg_pfx_z": "Pitch Out average vertical movement in inches. Variable, not relevant for performance analysis.",
  "SC_avg_pfx_z": "Screwball average vertical movement in inches. Typically negative (downward movement).",
  "SI_avg_pfx_z": "Sinker average vertical movement in inches. Typically near zero or slightly negative (less \"rise\" than four-seam, creates sink).",
  "SL_avg_pfx_z": "Slider average vertical movement in inches. Negative values (downward tilt, typically -1 to -5 inches).",
  "ST_avg_pfx_z": "Sweeper average vertical movement in inches. Near zero or slightly negative (more horizontal than traditional slider).",
  "SV_avg_pfx_z": "Slurve average vertical movement in inches. Moderate negative values (drop between slider and curve).",
  "UN_avg_pfx_z": "Unclassified pitch average vertical movement in inches. Variable depending on pitch type."
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

