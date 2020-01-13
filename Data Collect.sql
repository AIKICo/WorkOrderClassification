SELECT 
NeedDescription as v1,
CASE 
	WHEN Cons=1 AND HardWare=0 THEN N'CONS'
	ELSE N'NOTCONS'
END AS v1
FROM
(
	SELECT * From WorkOrder WHERE Cons=1 AND HardWare=0
	UNION
	SELECT TOP 4839 * From WorkOrder WHERE Cons=0 
) AS Ta ORDER BY NEWID()