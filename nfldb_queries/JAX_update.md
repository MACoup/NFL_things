# need to manually workaround the 'JAX' problem
```sql
insert into team values ('JAX','Jacksonville','Jaguars');
```

# verify
```sql
SELECT COUNT(*) from play where pos_team = 'JAX';
SELECT COUNT(*) from drive where pos_team = 'JAX';
SELECT COUNT(*) from game where home_team = 'JAX';
SELECT COUNT(*) from game where away_team = 'JAX';
SELECT COUNT(*) from player where team = 'JAX';
SELECT COUNT(*) from play_player where team = 'JAX';
```

# only play needs updating
```sql
UPDATE play SET pos_team = 'JAC' WHERE pos_team = 'JAX';
```

# delete 'JAX' from team
```sql
DELETE FROM team WHERE team_id = 'JAX';
```
