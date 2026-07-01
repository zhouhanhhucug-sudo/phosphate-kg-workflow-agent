// 1 跳邻域
MATCH (n:Entity {name: 'P2O5 含量'})-[r]-(m:Entity)
RETURN n, r, m
LIMIT 100;

// 1-3 跳邻域
MATCH path = (n:Entity {name: 'P2O5 含量'})-[*1..3]-(m:Entity)
RETURN path
LIMIT 200;

// 按关系类型统计邻域
MATCH (n:Entity {name: 'P2O5 含量'})-[r*1..3]-(m:Entity)
UNWIND r AS rel
RETURN type(rel) AS relation_type, count(*) AS relation_count
ORDER BY relation_count DESC;