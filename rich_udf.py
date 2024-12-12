from turboml.common.feature_engineering import TurboMLScalarFunction
from psycopg_pool import ConnectionPool


class PostgresLookup(TurboMLScalarFunction):
    def __init__(self, user, password, host, port, dbname):
        conninfo = (
            f"user={user} password={password} host={host} port={port} dbname={dbname}"
        )
        self.connPool = ConnectionPool(conninfo=conninfo)

    def func(self, index: str):
        with self.connPool.connection() as risingwaveConn:
            with risingwaveConn.cursor() as cur:
                query = 'SELECT "model_length" FROM r2dt_models WHERE id = %s'
                cur.execute(query, (index,))
                result = cur.fetchone()
        return result[0] if result else 0
