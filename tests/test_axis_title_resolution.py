from src.contracts import Box, Detection, DetectionFrame, OCRFrame, RecognizedText
from src.reconstruction.axis_titles import resolve_axis_titles


def test_inferred_axes_reassign_and_reject_titles():
    items = [
        Detection('bar1', 'vertical_bar', Box(40, 40, 50, 90), .9),
        Detection('bar2', 'vertical_bar', Box(70, 30, 80, 90), .9),
        Detection('yt1', 'y_tick_label', Box(20, 45, 30, 50), .9),
        Detection('yt2', 'y_tick_label', Box(20, 70, 30, 75), .9),
        Detection('xt1', 'x_tick_label', Box(40, 92, 50, 97), .9),
        Detection('xt2', 'x_tick_label', Box(70, 92, 80, 97), .9),
        Detection('title', 'x_axis_title', Box(2, 45, 15, 65), .9),
        Detection('legend', 'x_axis_title', Box(90, 45, 99, 65), .9),
    ]
    frame = DetectionFrame('test/chart.png', 100, 110, tuple(items))
    ocr = OCRFrame(frame.source_key, (RecognizedText('title', 'Number', .9),
                                      RecognizedText('legend', 'Adult visitors', .9)))
    result = resolve_axis_titles(frame, ocr)
    titles = [(d.id, d.role) for d in result.detections if d.role.endswith('axis_title')]
    assert titles == [('title', 'y_axis_title')]


def test_detected_plot_assigns_printed_titles_for_horizontal_chart():
    frame = DetectionFrame('test/chart.png', 100, 100, (
        Detection('plot', 'plot_area', Box(25, 10, 85, 75), .9),
        Detection('bottom', 'y_axis_title', Box(35, 85, 75, 95), .9),
        Detection('left', 'x_axis_title', Box(2, 30, 15, 60), .9),
    ))
    ocr = OCRFrame(frame.source_key, (RecognizedText('bottom', 'Value', .9),
                                      RecognizedText('left', 'Category', .9)))
    result = resolve_axis_titles(frame, ocr, orientation='horizontal')
    assert {d.id: d.role for d in result.detections if d.id != 'plot'} == {
        'bottom': 'x_axis_title', 'left': 'y_axis_title'}


def test_absent_title_stays_absent():
    frame = DetectionFrame('test/chart.png', 100, 100, (
        Detection('plot', 'plot_area', Box(25, 10, 85, 75), .9),
    ))
    assert resolve_axis_titles(frame, OCRFrame(frame.source_key, ())) == frame
