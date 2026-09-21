from src.plate_format import format_plate_display, parse_and_correct_plate, plates_similar


def test_mk_to_mm_correction():
    assert parse_and_correct_plate("KA 02 MK 9091") == "KA02MM9091"
    assert parse_and_correct_plate("KA02MK9091IN") == "KA02MM9091"
    assert parse_and_correct_plate("KA02MK9091.in") == "KA02MM9091"


def test_mn_plate():
    assert parse_and_correct_plate("KA 02 MN 1826") == "KA02MN1826"
    assert parse_and_correct_plate("KA02MN1826") == "KA02MN1826"


def test_display_format():
    assert format_plate_display("KA02MM9091") == "KA 02 MM 9091"


def test_trailing_s_to_five():
    assert parse_and_correct_plate("8254S") == "82545"
    assert parse_and_correct_plate("8254S IN") == "82545"
    assert parse_and_correct_plate("8254SIN") == "82545"


def test_display_numeric():
    assert format_plate_display("82545") == "82545"


def test_fuzzy_similarity():
    assert plates_similar("KA02MM9091", "KA02MK9091")
    assert not plates_similar("KA02MM9091", "KA02MM9092")


if __name__ == "__main__":
    test_mk_to_mm_correction()
    test_mn_plate()
    test_display_format()
    test_trailing_s_to_five()
    test_display_numeric()
    test_fuzzy_similarity()
    print("All plate format tests passed.")
