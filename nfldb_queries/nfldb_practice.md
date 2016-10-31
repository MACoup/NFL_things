# nfldb
public | agg_play    | table | nfldb
public | drive       | table | nfldb
public | game        | table | nfldb
public | meta        | table | nfldb
public | play        | table | nfldb
public | play_player | table | nfldb
public | player      | table | nfldb
public | team        | table | nfldb

```sql
SELECT full_name FROM player
WHERE team LIKE 'ATL';
```

```sql
SELECT week, day_of_week, start_time, home_team, home_score, away_team, away_score FROM game
WHERE season_year = 2016 AND week = 1 AND season_type='Regular';
```

```sql
SELECT game.season_year, game.week, player.full_name, player.position, play_player.receiving_tar, play_player.receiving_rec, play_player.receiving_yds FROM play_player
JOIN player ON player.player_id = play_player.player_id
JOIN game ON game.gsis_id = play_player.gsis_id
WHERE player.team = 'DEN'
AND play_player.receiving_tar > 0
ORDER BY game.season_year DESC, game.week ASC;
```
# Passing Points DF:
```sql
SELECT game.season_year, game.season_type, game.week, game.home_team, game.home_score, game.away_team, game.away_score, player.team, player.full_name, player.position, SUM(play_player.passing_att) AS passing_att, SUM(play_player.passing_cmp) AS passing_cmp, SUM(play_player.passing_yds) AS passing_yds,
SUM(play_player.passing_int) AS passing_int, SUM(play_player.passing_tds) AS passing_tds, SUM(play_player.passing_twopta) AS passing_twopta, SUM(play_player.passing_twoptm) AS passing_twoptm, SUM(play_player.receiving_rec) AS receiving_rec, SUM(play_player.receiving_tar) AS receiving_tar, SUM(play_player.receiving_tds) AS receiving_tds, SUM(play_player.receiving_twopta) AS receiving_twopta, SUM(play_player.receiving_twoptm) AS receiving_twoptm, SUM(play_player.receiving_yac_yds) AS receiving_yac, SUM(play_player.receiving_yds) AS receiving_yds, SUM(play_player.rushing_att) AS rushing_att, SUM(play_player.rushing_yds) AS rushing_yds, SUM(play_player.rushing_tds) AS rushing_tds, SUM(play_player.rushing_loss_yds) AS rushing_loss_yards, SUM(play_player.rushing_twoptm) AS rushing_twoptm, SUM(play_player.fumbles_tot) AS fumbles_total, SUM(play_player.fumbles_rec_tds) AS fumble_rec_tds FROM play_player JOIN player ON player.player_id = play_player.player_id JOIN game ON game.gsis_id = play_player.gsis_id WHERE player.position = 'QB' GROUP BY game.season_year, game.season_type, game.week, game.home_team, game.home_score, game.away_team, game.away_score, player.team, player.full_name, player.position HAVING SUM(play_player.passing_att) > 0 ORDER BY game.season_year DESC, game.week ASC;
```
#Receiving Points DF:
```sql
SELECT game.season_year, game.season_type, game.week, game.home_team, game.home_score, game.away_team, game.away_score, player.team, player.full_name, player.position, SUM(play_player.receiving_rec) AS receiving_rec, SUM(play_player.receiving_tar) AS receiving_tar, SUM(play_player.receiving_tds) AS receiving_tds, SUM(play_player.receiving_twopta) AS receiving_twopta, SUM(play_player.receiving_twoptm) AS receiving_twoptm, SUM(play_player.receiving_yac_yds) AS receiving_yac, SUM(play_player.receiving_yds) AS receiving_yds, SUM(play_player.fumbles_tot) AS fumbles_total, SUM(play_player.fumbles_rec_tds) AS fumble_rec_tds, SUM(play_player.kicking_rec_tds) AS kicking_rec_tds, SUM(play_player.puntret_tds) AS punt_ret_tds, SUM(play_player.rushing_att) AS rushing_att, SUM(play_player.rushing_yds) AS rushing_yds, SUM(play_player.rushing_tds) AS rushing_tds, SUM(play_player.rushing_loss_yds) AS rushing_loss_yards, SUM(play_player.rushing_twoptm) AS rushing_twoptm FROM play_player JOIN player ON player.player_id = play_player.player_id JOIN game on game.gsis_id = play_player.gsis_id WHERE player.position = 'WR' GROUP BY game.season_year, game.season_type, game.week, game.home_team, game.home_score, game.away_team, game.away_score, player.team, player.full_name, player.position HAVING SUM(play_player.receiving_rec) > 0 ORDER BY game.season_year DESC, game.week ASC;
```

#Rushing Points DF:
```sql
SELECT game.season_year, game.season_type, game.week, game.home_team, game.home_score, game.away_team, game.away_score, player.team, player.full_name, player.position, SUM(play_player.receiving_rec) AS receiving_rec, SUM(play_player.receiving_tar) AS receiving_tar, SUM(play_player.receiving_tds) AS receiving_tds, SUM(play_player.receiving_twopta) AS receiving_twopta, SUM(play_player.receiving_twoptm) AS receiving_twoptm, SUM(play_player.receiving_yac_yds) AS receiving_yac, SUM(play_player.receiving_yds) AS receiving_yds, SUM(play_player.fumbles_tot) AS fumbles_total, SUM(play_player.fumbles_rec_tds) AS fumble_rec_tds, SUM(play_player.kicking_rec_tds) AS kicking_rec_tds, SUM(play_player.puntret_tds) AS punt_ret_tds, SUM(play_player.rushing_att) AS rushing_att, SUM(play_player.rushing_yds) AS rushing_yds, SUM(play_player.rushing_tds) AS rushing_tds, SUM(play_player.rushing_loss_yds) AS rushing_loss_yards, SUM(play_player.rushing_twoptm) AS rushing_twoptm FROM play_player JOIN player ON player.player_id = play_player.player_id JOIN game on game.gsis_id = play_player.gsis_id WHERE player.position = 'RB' GROUP BY game.season_year, game.season_type, game.week, game.home_team, game.home_score, game.away_team, game.away_score, player.team, player.full_name, player.position HAVING SUM(play_player.receiving_rec) > 0 ORDER BY game.season_year DESC, game.week ASC;
```

# Tight End Points DF
```sql
SELECT game.season_year, game.season_type, game.week, game.home_team, game.home_score, game.away_team, game.away_score, player.team, player.full_name, player.position, SUM(play_player.receiving_rec) AS receiving_rec, SUM(play_player.receiving_tar) AS receiving_tar, SUM(play_player.receiving_tds) AS receiving_tds, SUM(play_player.receiving_twopta) AS receiving_twopta, SUM(play_player.receiving_twoptm) AS receiving_twoptm, SUM(play_player.receiving_yac_yds) AS receiving_yac, SUM(play_player.receiving_yds) AS receiving_yds, SUM(play_player.fumbles_tot) AS fumbles_total, SUM(play_player.fumbles_rec_tds) AS fumble_rec_tds, SUM(play_player.kicking_rec_tds) AS kicking_rec_tds, SUM(play_player.puntret_tds) AS punt_ret_tds, SUM(play_player.rushing_att) AS rushing_att, SUM(play_player.rushing_yds) AS rushing_yds, SUM(play_player.rushing_tds) AS rushing_tds, SUM(play_player.rushing_loss_yds) AS rushing_loss_yards, SUM(play_player.rushing_twoptm) AS rushing_twoptm FROM play_player JOIN player ON player.player_id = play_player.player_id JOIN game on game.gsis_id = play_player.gsis_id WHERE player.position = 'TE' GROUP BY game.season_year, game.season_type, game.week, game.home_team, game.home_score, game.away_team, game.away_score, player.team, player.full_name, player.position HAVING SUM(play_player.receiving_rec) > 0 ORDER BY game.season_year DESC, game.week ASC;
```
# Defense Points DF
```sql
SELECT game.season_year, game.season_type, game.week, game.home_team, game.home_score, game.away_team, game.away_score, play_player.team, SUM(play_player.defense_ffum) AS forced_fumble, SUM(play_player.defense_fgblk) AS fg_block, SUM(play_player.defense_frec) AS fumble_rec, SUM(play_player.defense_frec_tds) AS fumble_rec_tds, SUM(play_player.defense_int) AS ints, SUM(play_player.defense_int_tds) AS int_tds, SUM(play_player.defense_misc_tds) AS misc_tds, SUM(play_player.defense_puntblk) AS punt_block, SUM(play_player.defense_safe) AS safety, SUM(play_player.defense_sk) AS sack, SUM(play_player.defense_xpblk) AS xp_block FROM play_player JOIN game ON play_player.gsis_id = game.gsis_id JOIN player ON play_player.player_id = player.player_id GROUP BY game.season_year, game.season_type, game.week, game.home_team, game.home_score, game.away_team, game.away_score, play_player.team ORDER BY game.season_year DESC, game.week ASC;
```

# Passing Stats DF
# Defense agg_play
```sql
SELECT agg_play.defense_ffum, play.pos_team FROM agg_play
JOIN play ON agg_play.play_id = play.play_id
WHERE defense_ffum > 0;
```

# More Defense practice
```sql
SELECT agg_play.defense_ffum, game.home_team, game.away_team, play.pos_team
FROM agg_play JOIN game on agg_play.gsis_id = game.gsis_id
JOIN play on agg_play.gsis_id = play.gsis_id
WHERE game.season_year = 2016 AND agg_play.defense_ffum > 0;
```



```sql
SELECT player.full_name, SUM(play_player.passing_yds) AS passing_yds
FROM player
JOIN play_player ON player.player_id = play_player.player_id
JOIN game ON game.gsis_id = play_player.gsis_id
WHERE game.season_year =  2016 AND game.season_type = 'Regular'
GROUP BY player.full_name
HAVING SUM(play_player.passing_yds) > 0
ORDER BY  passing_yds DESC;
```

```sql
SELECT player.full_name, SUM(play_player.fumbles_lost) AS fumbles_lost
FROM player
JOIN play_player ON player.player_id = play_player.player_id
JOIN game ON game.gsis_id = play_player.gsis_id
WHERE game.season_year = 2016 AND game.season_type = 'Regular'
GROUP BY player.full_name
HAVING SUM(play_player.fumbles_lost) > 0
ORDER BY fumbles_lost;
```

```sql
SELECT
```
