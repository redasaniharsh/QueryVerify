"""
Unit tests for schema-aware starter question generation and edge cases.
"""

from frontend.starter_questions import (
    classify_columns,
    generate_starter_questions,
    get_active_schema_tables,
)


def test_classify_columns_basic():
    cols = [
        ("id", "INTEGER"),
        ("customer_id", "BIGINT"),
        ("name", "VARCHAR"),
        ("country", "TEXT"),
        ("sales_amount", "DECIMAL"),
        ("order_date", "DATE"),
    ]
    res = classify_columns(cols)
    assert "sales_amount" in res["numeric"]
    assert "country" in res["text"]
    assert "name" in res["text"]
    assert "order_date" in res["date"]
    # id and customer_id must not be treated as summable numeric metrics
    assert "id" not in res["numeric"]
    assert "customer_id" not in res["numeric"]


def test_empty_schema():
    assert generate_starter_questions([]) == []


def test_single_table_two_text_columns_no_numeric():
    """Edge case 1: 1 table, 2 text columns, no numeric columns."""
    tables = [
        {
            "table": "tags",
            "numeric": [],
            "text": ["tag_name", "author"],
            "date": [],
            "all": ["tag_name", "author"],
        }
    ]
    questions = generate_starter_questions(tables)
    assert len(questions) > 0
    assert len(questions) <= 4
    for q in questions:
        assert isinstance(q, str)
        assert "None" not in q
        assert "tags" in q
    assert "How many rows are in tags?" in questions


def test_single_table_one_column():
    """Edge case: 1 table, 1 column."""
    tables = [
        {
            "table": "inventory",
            "numeric": [],
            "text": ["item_name"],
            "date": [],
            "all": ["item_name"],
        }
    ]
    questions = generate_starter_questions(tables)
    assert len(questions) >= 1
    assert "How many rows are in inventory?" in questions
    for q in questions:
        assert "None" not in q


def test_numeric_only_schema():
    """Edge case: table with only numeric columns."""
    tables = [
        {
            "table": "measurements",
            "numeric": ["temperature", "humidity"],
            "text": [],
            "date": [],
            "all": ["temperature", "humidity"],
        }
    ]
    questions = generate_starter_questions(tables)
    assert len(questions) >= 2
    assert "How many rows are in measurements?" in questions
    for q in questions:
        assert "None" not in q


def test_titanic_schema_generation():
    """Verify Titanic dataset question generation."""
    tables = [
        {
            "table": "titanic",
            "numeric": ["Age", "Fare", "SibSp", "Parch"],
            "text": ["Pclass", "Sex", "Embarked"],
            "date": [],
            "all": ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
        }
    ]
    questions = generate_starter_questions(tables)
    assert len(questions) == 4
    for q in questions:
        assert "None" not in q
        assert "titanic" in q
    # Check that age/fare and pclass/embarked are featured
    all_text = " ".join(questions)
    assert "rows" in all_text
    assert ("Age" in all_text or "Fare" in all_text)
    assert ("Pclass" in all_text or "Embarked" in all_text)


def test_sample_schema_generation():
    """Verify default sample schema question generation."""
    tables = get_active_schema_tables(db_mode="Sample Data")
    assert len(tables) >= 1
    questions = generate_starter_questions(tables)
    assert len(questions) == 4
    for q in questions:
        assert "None" not in q
        assert len(q) > 10
