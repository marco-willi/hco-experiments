################################################################
# Test functions for user agent class

from pprint import pprint
from swap.agents.user import User
from swap.agents.bureau import Bureau
from swap.agents.tracker import *
from swap.swap import Classification

from unittest.mock import patch, MagicMock

# Sample classification
cl = Classification.generate({
    'user_name': 'HTMAMR38',
    'metadata': {'mag_err': 0.1, 'mag': 20.666},
    'gold_label': 0,
    'diff': '1172057211575001100_57535.517_76128180_554_diff.jpeg',
    'object_id': '1172057211575001100',
    'classification_id': 13216944,
    'annotation': 1,
    'subject_id': 2149031,
    'machine_score': 0.960535,
    'user_id': 1497743})
uid = 0
epsilon = .5


class TestUser:
    def test_init(self):
        u = User(uid, epsilon)

        assert type(u.annotations) is Tracker
        assert type(u.gold_labels) is Tracker

        assert type(u.trackers) is Tracker_Collection

    @patch.object(Tracker, 'add')
    def test_addcl_updates_annotation_gold_labels(self, mock):
        u = User(uid, epsilon)
        u.annotations.add = MagicMock()
        u.gold_labels.add = MagicMock()
        u.trackers.trackers[0].add = MagicMock()

        u.addClassification(cl, cl.gold)
        u.annotations.add.assert_called_once_with(1)
        u.gold_labels.add.assert_called_once_with(0)
        u.trackers.trackers[0].add.assert_called_once_with(1)

    def test_score_all_correct(self):
        u = User(uid, epsilon)
        data = [
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1)
        ]

        for g, a in data:
            u.addClassification(Classification(0, 0, a), g)

        assert u.getScore(1) == 5 / 6

    def test_score_half_correct(self):
        u = User(uid, epsilon)
        data = [
            (1, 1),
            (1, 1),
            (1, 0),
            (1, 0)
        ]

        for g, a in data:
            u.addClassification(Classification(0, 0, a), g)

        assert u.getScore(1) == 3 / 6

    def test_score_0_correct(self):
        u = User(uid, epsilon)
        data = [
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 0)
        ]

        for g, a in data:
            u.addClassification(Classification(0, 0, a), g)

        assert u.getScore(1) == 1 / 6

    def test_export_score0(self):
        u = User(uid, epsilon)
        data = [
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (0, 0),
            (0, 0),
            (0, 1),
            (0, 1)
        ]

        for g, a in data:
            u.addClassification(Classification(0, 0, a), g)

        export = u.export()
        pprint(export)
        assert export['score_0'] == .5

    def test_export_score1(self):
        u = User(uid, epsilon)
        data = [
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (0, 0),
            (0, 0),
            (0, 1),
            (0, 1)
        ]

        for g, a in data:
            u.addClassification(Classification(0, 0, a), g)

        export = u.export()
        pprint(export)
        assert export['score_1'] == .5

    def test_export_gold_labels(self):
        u = User(uid, epsilon)
        data = [
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (0, 0),
            (0, 0),
            (0, 1),
            (0, 1)
        ]

        for g, a in data:
            u.addClassification(Classification(0, 0, a), g)

        export = u.export()
        pprint(export)
        assert export['gold_labels'] == [1, 1, 1, 1, 0, 0, 0, 0]

    def test_export_score0_history(self):
        u = User(uid, epsilon)
        data = [
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (0, 0),
            (0, 0),
            (0, 1),
            (0, 1)
        ]

        for g, a in data:
            u.addClassification(Classification(0, 0, a), g)

        export = u.export()
        pprint(export)
        assert export['score_0_history'] == [.5, 2/3, 3/4, 3/5, 3/6]

    def test_export_score1_history(self):
        u = User(uid, epsilon)
        data = [
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (0, 0),
            (0, 0),
            (0, 1),
            (0, 1)
        ]

        for g, a in data:
            u.addClassification(Classification(0, 0, a), g)

        export = u.export()
        pprint(export)
        assert export['score_1_history'] == [.5, 1 / 3, 1 / 4, 2 / 5, 3 / 6]

    def test_bureau_stats(self):
        b = Bureau(User)
        b.add(User(0, .5))
        b.add(User(1, .25))
        b.add(User(2, .75))

        s = b.stats()
        assert 0 in s.stats
        assert 1 in s.stats
        assert len(s.stats) == 2
