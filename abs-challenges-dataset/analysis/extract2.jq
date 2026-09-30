# One row per ABS pitch challenge (reviewType "MJ").
# Two places hold the review in StatsAPI feed/live:
#   - on the pitch event (playEvents[i].reviewDetails) when the plate appearance goes on afterwards;
#   - on the play itself (allPlays[k].reviewDetails) when the pitch, as finally called, ends the plate
#     appearance (strikeout / walk). Then the challenged pitch is the last pitch of the play.
# Count, outs and score are taken from BEFORE the challenged pitch.
# NOTE: `call` is the FINAL call after the review -- never show it to a model; derive the original call
# from which side challenged.
.gameData as $g
| .liveData.plays.allPlays as $plays
| range(0; $plays | length) as $k
| $plays[$k] as $p
| ($p.playEvents) as $ev
| (
    ( range(0; $ev | length) as $i | select($ev[$i].reviewDetails.reviewType == "MJ")
      | {i: $i, rv: $ev[$i].reviewDetails, where: "pitch"} ),
    ( select($p.reviewDetails.reviewType == "MJ")
      | ([range(0; $ev | length) | select($ev[.].isPitch == true)] | last) as $li
      | select($li != null)
      | {i: $li, rv: $p.reviewDetails, where: "play"} )
  ) as $h
| $ev[$h.i] as $e
| ([$ev[0:$h.i][] | select(.count != null)] | last | .count) as $prev
| (if $k == 0 then {homeScore: 0, awayScore: 0} else $plays[$k - 1].result end) as $sc
| {
    date: $date, gamePk: $g.game.pk, inning: $p.about.inning, half: $p.about.halfInning,
    home: $g.teams.home.abbreviation, away: $g.teams.away.abbreviation,
    home_id: $g.teams.home.id, away_id: $g.teams.away.id,
    home_score_before: $sc.homeScore, away_score_before: $sc.awayScore,
    balls: ($prev.balls // 0), strikes: ($prev.strikes // 0), outs: ($prev.outs // ($e.count.outs)),
    call: $e.details.call.description, pitch_type: $e.details.type.code, speed: $e.pitchData.startSpeed,
    pX: $e.pitchData.coordinates.pX, pZ: $e.pitchData.coordinates.pZ,
    sz_top: $e.pitchData.strikeZoneTop, sz_bot: $e.pitchData.strikeZoneBottom,
    overturned: $h.rv.isOverturned, challenge_team: $h.rv.challengeTeamId,
    challenger_id: $h.rv.player.id, challenger: $h.rv.player.fullName,
    batter_id: $p.matchup.batter.id, pitcher_id: $p.matchup.pitcher.id,
    batting_team_is_home: ($p.about.halfInning == "bottom"),
    review_on: $h.where, play_event: $p.result.event
  }
