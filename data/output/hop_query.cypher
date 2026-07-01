MATCH path = (core:Entity {name: 'P2O5 含量'})-[*1..3]-(neighbor:Entity)
RETURN path, length(path) AS hop_count
ORDER BY hop_count ASC
LIMIT 200;