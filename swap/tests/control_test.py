#!/usr/bin/env python
################################################################
# Script to test control functionality

import swap.db.classifications as dbcl
from swap.control import Control
from swap.control import GoldGetter
from unittest.mock import MagicMock
import pytest

fields = {'user_id', 'classification_id', 'subject_id',
          'annotation', 'gold_label'}


# def test_classifications_projection():
#     q = Query()
#     q.fields(['user_id', 'classification_id'])
#     raw = dbcl.classifications.aggregate(q.build())

#     item = raw.next()

#     assert 'user_id' in item
#     assert 'classification_id' in item
#     assert 'subject_id' not in item


# def test_classifications_limit():
#     q = Query()
#     q.fields(fields).limit(5)
#     raw = dbcl.classifications.aggregate(q.build())

#     assert len(list(raw)) == 5


# def test_users():
#     control = swap.Server(.5, .5)
#     users = control.getClassificationsByUser()

#     pprint(list(users))


@pytest.mark.skip(reason='Takes too long')
def test_get_one_classification():
    """ Get the first classification
    """
    control = Control(0.5, 0.5)

    cursor = control.getClassifications()
    n_class = len(cursor)
    cl = cursor.next()

    assert n_class > 0
    assert type(cl) == dict
    assert len(cl) > 0


def test_with_train_split():
    old = dbcl.getRandomGoldSample
    mock = MagicMock(return_value=[])
    dbcl.getRandomGoldSample = mock

    c = Control(.5, .5, mock)
    c.gold_getter.random(100)
    c.getGoldLabels()

    mock.assert_called_with(100, type_=tuple)

    dbcl.getRandomGoldSample = old


def test_without_train_split():
    old = dbcl.getAllGolds
    mock = MagicMock(return_value={})
    dbcl.getAllGolds = mock

    c = Control(.5, .5, mock)
    c.getGoldLabels()

    mock.assert_called_with(type_=tuple)

    dbcl.getAllGolds = old


class TestGoldGetter:

    def test_wrapper_golds_to_None(self):
        old = dbcl.getAllGolds
        dbcl.getAllGolds = MagicMock(return_value=[])

        gg = GoldGetter()
        gg._golds = {}
        gg.all()

        assert gg._golds is None

        dbcl.getAllGolds = old

    def test_wrapper_getter(self):
        old = dbcl.getAllGolds
        dbcl.getAllGolds = MagicMock(return_value=[])

        gg = GoldGetter()
        gg._golds = {}
        gg.all()

        print(type(gg.getter))
        assert callable(gg.getter)

        dbcl.getAllGolds = old
