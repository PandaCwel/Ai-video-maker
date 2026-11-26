from app.services.director import _chunks


def test_long_screenplay_is_chunked_without_losing_text():
    scenes = [f"INT. ROOM {index} - NIGHT\n\nA character says line {index}.\n" for index in range(1, 80)]
    screenplay = "\n".join(scenes)
    chunks = _chunks(screenplay, limit=900)
    assert len(chunks) > 1
    assert "".join(chunks).replace("\n", "").replace(" ", "") == screenplay.replace("\n", "").replace(" ", "")
    assert "A character says line 79." in "".join(chunks)
