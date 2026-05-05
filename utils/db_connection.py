"""
Database connection utilities for XDSData Dashboard
Using PyMySQL (pure Python, no compilation needed)
"""

import pymysql
from pymysql import Error
import streamlit as st
import logging
from config.settings import DB_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    _instance = None
    _connection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance

    def connect(self):
        try:
            if self._connection is None or not self._connection.open:
                self._connection = pymysql.connect(**DB_CONFIG)
                logger.info("Database connection established")
            return self._connection
        except Error as e:
            logger.error(f"Database connection failed: {e}")
            return None

    def disconnect(self):
        if self._connection and self._connection.open:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    def execute_query(self, query, params=None):
        connection = self.connect()
        if not connection:
            return []

        cursor = connection.cursor(pymysql.cursors.DictCursor)
        try:
            cursor.execute(query, params or ())
            result = cursor.fetchall()
            return result
        except Error as e:
            logger.error(f"Query execution failed: {e}")
            return []
        finally:
            cursor.close()

    def execute_insert(self, query, params=None):
        connection = self.connect()
        if not connection:
            return None

        cursor = connection.cursor()
        try:
            cursor.execute(query, params or ())
            connection.commit()
            return cursor.lastrowid
        except Error as e:
            logger.error(f"Insert failed: {e}")
            connection.rollback()
            return None
        finally:
            cursor.close()


db = DatabaseConnection()


@st.cache_resource
def get_db():
    """Get database connection (cached for Streamlit)"""
    return db
