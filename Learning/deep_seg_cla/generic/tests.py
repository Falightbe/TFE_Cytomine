from unittest import TestCase

import numpy as np

from generic.dataset import ImageDataset, ImageClassificationDataset


class TestImageClassificationDataset(TestCase):

    def testNoDirs(self):
        path = "C:/data/cytomine/thyroid/raw"
        dataset1 = ImageClassificationDataset(path)
        self.assertEqual(dataset1.size(), 6326)
        x, y = dataset1.all()
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertSetEqual(set(np.unique(y)), {
            15109451, 15109483, 15109489, 15109495, 22042230, 675999,
            676026, 676176, 676210, 676390, 676407, 676434, 676446,
            8844845, 8844862, 933004
        })

        dataset2 = ImageClassificationDataset(path, encode_classes=True)
        self.assertEqual(dataset2.size(), 6326)
        x, y = dataset2.all()
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertSetEqual(set(np.unique(y)), set(range(16)))

        x, y = dataset2.all(classes=[675999, 676026])
        self.assertEqual(x.shape[0], 1557)
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertSetEqual(set(np.unique(y)), {0, 1})

        dataset3 = ImageClassificationDataset(path)
        x, y = dataset3.all(classes=[675999, 676026])
        self.assertEqual(x.shape[0], 1557)
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertSetEqual(set(np.unique(y)), {675999, 676026})

    def testDirs(self):
        path = "C:/data/cytomine/thyroid/augmented/patterns_noresize"
        dataset1 = ImageClassificationDataset(path, dirs=["train", "test"])
        self.assertEqual(dataset1.size(), 11547)
        self.assertEqual(dataset1.size(dirs=["train"]), 11036)
        self.assertEqual(dataset1.size(dirs=["test"]), 511)
        x, y = dataset1.all(dirs=["test"])
        self.assertSetEqual(set(np.unique(y)), {0, 1})
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertEqual(x.shape[0], 511)


class TestImageDataset(TestCase):

    def testOneHotEncoding1D(self):
        array = np.array([1, 2, 3, 5, 5, 2, 3, 1, 1, 5, 2])
        encoded = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0]
        ])

        with_classes = ImageDataset.one_hot_encode(array, classes=[1, 2, 3, 5])
        without_classes = ImageDataset.one_hot_encode(array)
        np.testing.assert_array_almost_equal(with_classes, encoded)
        np.testing.assert_array_almost_equal(without_classes, encoded)

    def testOneHotEncoding2D(self):
        array = np.array([
            [1, 0, 0, 1, 1],
            [0, 1, 2, 0, 1]
        ])
        encoded = np.array([
            [
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0]
            ],
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ]
        ])
        with_classes = ImageDataset.one_hot_encode(array, classes=[0, 1, 2])
        without_classes = ImageDataset.one_hot_encode(array)
        np.testing.assert_array_almost_equal(with_classes, encoded)
        np.testing.assert_array_almost_equal(without_classes, encoded)

    def testOneHotEncodingUnknownClasses(self):
        array = np.array([1, 2, 3, 5, 5, 2, 3, 1, 1, 5, 2])
        encoded = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        with_larger = ImageDataset.one_hot_encode(array, classes=[1, 2, 3, 6])
        np.testing.assert_array_almost_equal(with_larger, encoded)
        with self.assertRaises(ValueError):
            _ = ImageDataset.one_hot_encode(array, classes=[1, 2, 3, 4])

    def testMakeBinary1D(self):
        array = np.array([2, 5, 2, 2, 5, 3])
        binary = np.array([0, 1, 0, 0, 1, 1])
        np.testing.assert_array_almost_equal(ImageDataset.make_binary(array, positive=[5, 3]), binary)

    def testMakeBinary2D(self):
        array = np.array([
            [2, 5, 2, 2, 5, 3],
            [1, 2, 5, 3, 3, 5]
        ])
        binary = np.array([
            [0, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 1]
        ])
        np.testing.assert_array_almost_equal(ImageDataset.make_binary(array, positive=[5, 3]), binary)