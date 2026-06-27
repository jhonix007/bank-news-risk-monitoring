import pytest

from app.services.news_file_service import read_and_validate_csv


def test_csv_without_title_is_rejected():
    with pytest.raises(ValueError, match="title"):
        read_and_validate_csv("text_fragment\nТекст\n".encode("utf-8"))


def test_csv_without_text_fragment_is_rejected():
    with pytest.raises(ValueError, match="text_fragment"):
        read_and_validate_csv("title\nЗаголовок\n".encode("utf-8"))


def test_csv_without_entity_norm_is_accepted():
    frame = read_and_validate_csv("title,text_fragment\nЗаголовок,Текст\n".encode("utf-8"))
    assert len(frame) == 1


def test_empty_csv_is_rejected():
    with pytest.raises(ValueError, match="пустой"):
        read_and_validate_csv(b"")


def test_valid_csv_with_extra_columns_is_accepted():
    data = "source,url,published_at,title,text_fragment,entity_norm,extra\nРБК,u,2026,Новость,Текст,Сбербанк,x\n"
    frame = read_and_validate_csv(data.encode("utf-8"))
    assert len(frame) == 1
