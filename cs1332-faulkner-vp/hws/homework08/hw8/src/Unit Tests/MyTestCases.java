import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;
import static org.junit.Assert.assertThrows;

public class MyTestCases {

    private static final int TIMEOUT = 200;
    private IntegerWrapper[] integers;
    private IntegerWrapper[] sorted;
    private ComparatorPlus<IntegerWrapper> comp;

    @Before
    public void setUp() {
        comp = IntegerWrapper.getComparator();
    }

    //Class for testing proper sorting.
    private static class IntegerWrapper {
        private Integer value;
        public IntegerWrapper(Integer value) {
            this.value = value;
        }
        public Integer getValue() {
            return value;
        }
        public void setValue(Integer value) {
            this.value = value;
        }
        public static ComparatorPlus<IntegerWrapper> getComparator() {
            return new ComparatorPlus<>() {
                @Override
                public int compare(IntegerWrapper int1,
                                   IntegerWrapper int2) {
                    incrementCount();
                    return int1.getValue().compareTo(int2.getValue());
                }
            };
        }
        @Override
        public String toString() {
            return "Value: " + value;
        }
        @Override
        public boolean equals(Object other) {
            if (other == null) {
                return false;
            }
            if (this == other) {
                return true;
            }
            return other instanceof IntegerWrapper
                    && ((IntegerWrapper) other).value.equals(this.value);
        }
    }

    //Class that allows for comparisons to be tracked
    private abstract static class ComparatorPlus<T> implements Comparator<T> {
        private int count;
        public int getCount() {
            return count;
        }
        public void incrementCount() {
            count++;
        }
    }

    @Test(timeout = TIMEOUT)
    public void testSelectionSortException() {
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.selectionSort(null, comp);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testSelectionSortException2() {
        integers = new MyTestCases.IntegerWrapper[8];
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.selectionSort(integers, null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testSelectionSort1() {
        integers = new MyTestCases.IntegerWrapper[10];
        integers[0] = new MyTestCases.IntegerWrapper(3);
        integers[1] = new MyTestCases.IntegerWrapper(-1);
        integers[2] = new MyTestCases.IntegerWrapper(0);
        integers[3] = new MyTestCases.IntegerWrapper(4);
        integers[4] = new MyTestCases.IntegerWrapper(-3);
        integers[5] = new MyTestCases.IntegerWrapper(1);
        integers[6] = new MyTestCases.IntegerWrapper(-2);
        integers[7] = new MyTestCases.IntegerWrapper(-4);
        integers[8] = new MyTestCases.IntegerWrapper(5);
        integers[9] = new MyTestCases.IntegerWrapper(2);

        sorted = new MyTestCases.IntegerWrapper[10];
        sorted[0] = new MyTestCases.IntegerWrapper(-4);
        sorted[1] = new MyTestCases.IntegerWrapper(-3);
        sorted[2] = new MyTestCases.IntegerWrapper(-2);
        sorted[3] = new MyTestCases.IntegerWrapper(-1);
        sorted[4] = new MyTestCases.IntegerWrapper(0);
        sorted[5] = new MyTestCases.IntegerWrapper(1);
        sorted[6] = new MyTestCases.IntegerWrapper(2);
        sorted[7] = new MyTestCases.IntegerWrapper(3);
        sorted[8] = new MyTestCases.IntegerWrapper(4);
        sorted[9] = new MyTestCases.IntegerWrapper(5);

        Sorting.selectionSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 45 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testSelectionSort2() {
        integers = new MyTestCases.IntegerWrapper[2];
        integers[0] = new MyTestCases.IntegerWrapper(0);
        integers[1] = new MyTestCases.IntegerWrapper(-1);

        sorted = new MyTestCases.IntegerWrapper[2];
        sorted[0] = new MyTestCases.IntegerWrapper(-1);
        sorted[1] = new MyTestCases.IntegerWrapper(0);

        Sorting.selectionSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() == 1);
    }

    @Test(timeout = TIMEOUT)
    public void testSelectionSortAlreadySorted() {
        integers = new MyTestCases.IntegerWrapper[10];
        integers[0] = new MyTestCases.IntegerWrapper(-4);
        integers[1] = new MyTestCases.IntegerWrapper(-3);
        integers[2] = new MyTestCases.IntegerWrapper(-2);
        integers[3] = new MyTestCases.IntegerWrapper(-1);
        integers[4] = new MyTestCases.IntegerWrapper(0);
        integers[5] = new MyTestCases.IntegerWrapper(1);
        integers[6] = new MyTestCases.IntegerWrapper(2);
        integers[7] = new MyTestCases.IntegerWrapper(3);
        integers[8] = new MyTestCases.IntegerWrapper(4);
        integers[9] = new MyTestCases.IntegerWrapper(5);

        sorted = new MyTestCases.IntegerWrapper[10];
        sorted[0] = new MyTestCases.IntegerWrapper(-4);
        sorted[1] = new MyTestCases.IntegerWrapper(-3);
        sorted[2] = new MyTestCases.IntegerWrapper(-2);
        sorted[3] = new MyTestCases.IntegerWrapper(-1);
        sorted[4] = new MyTestCases.IntegerWrapper(0);
        sorted[5] = new MyTestCases.IntegerWrapper(1);
        sorted[6] = new MyTestCases.IntegerWrapper(2);
        sorted[7] = new MyTestCases.IntegerWrapper(3);
        sorted[8] = new MyTestCases.IntegerWrapper(4);
        sorted[9] = new MyTestCases.IntegerWrapper(5);

        Sorting.selectionSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 45 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testSelectionSortSingleElement() {
        integers = new MyTestCases.IntegerWrapper[1];
        integers[0] = new MyTestCases.IntegerWrapper(0);

        sorted = new MyTestCases.IntegerWrapper[1];
        sorted[0] = new MyTestCases.IntegerWrapper(0);

        Sorting.selectionSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() == 0);
    }

    @Test(timeout = TIMEOUT)
    public void testCocktailSortException() {
        integers = new MyTestCases.IntegerWrapper[8];
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.cocktailSort(integers, null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testCocktailSortException2() {
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.cocktailSort(null, comp);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testCocktailSort1() {
        integers = new MyTestCases.IntegerWrapper[5];
        for (int i = 0; i < 5; i++) {
            integers[i] = new MyTestCases.IntegerWrapper(4 - i);
        }

        sorted = new MyTestCases.IntegerWrapper[5];
        for (int i = 0; i < 5; i++) {
            sorted[i] = new MyTestCases.IntegerWrapper(i);
        }

        Sorting.cocktailSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 10 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testCocktailSort2() {
        IntegerWrapper a = new IntegerWrapper(2);
        IntegerWrapper b = new IntegerWrapper(2);
        IntegerWrapper c = new IntegerWrapper(1);
        integers = new MyTestCases.IntegerWrapper[3];
        integers[0] = a;
        integers[1] = b;
        integers[2] = c;

        sorted = new MyTestCases.IntegerWrapper[3];
        sorted[0] = c;
        sorted[1] = a;
        sorted[2] = b;

        Sorting.cocktailSort(integers, comp);
        assertArrayEquals(sorted, integers);
        for (int i = 0; i < integers.length; i++) {
            //Tests stability
            assertTrue(integers[i] == sorted[i]);
        }

        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 3 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testCocktailSort3() {
        integers = new MyTestCases.IntegerWrapper[10];
        integers[0] = new MyTestCases.IntegerWrapper(3);
        integers[1] = new MyTestCases.IntegerWrapper(-1);
        integers[2] = new MyTestCases.IntegerWrapper(0);
        integers[3] = new MyTestCases.IntegerWrapper(4);
        integers[4] = new MyTestCases.IntegerWrapper(-3);
        integers[5] = new MyTestCases.IntegerWrapper(1);
        integers[6] = new MyTestCases.IntegerWrapper(-2);
        integers[7] = new MyTestCases.IntegerWrapper(-4);
        integers[8] = new MyTestCases.IntegerWrapper(5);
        integers[9] = new MyTestCases.IntegerWrapper(2);

        sorted = new MyTestCases.IntegerWrapper[10];
        sorted[0] = new MyTestCases.IntegerWrapper(-4);
        sorted[1] = new MyTestCases.IntegerWrapper(-3);
        sorted[2] = new MyTestCases.IntegerWrapper(-2);
        sorted[3] = new MyTestCases.IntegerWrapper(-1);
        sorted[4] = new MyTestCases.IntegerWrapper(0);
        sorted[5] = new MyTestCases.IntegerWrapper(1);
        sorted[6] = new MyTestCases.IntegerWrapper(2);
        sorted[7] = new MyTestCases.IntegerWrapper(3);
        sorted[8] = new MyTestCases.IntegerWrapper(4);
        sorted[9] = new MyTestCases.IntegerWrapper(5);

        Sorting.cocktailSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 34 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testCocktailSortAlreadySorted() {
        integers = new MyTestCases.IntegerWrapper[10];
        integers[0] = new MyTestCases.IntegerWrapper(-4);
        integers[1] = new MyTestCases.IntegerWrapper(-3);
        integers[2] = new MyTestCases.IntegerWrapper(-2);
        integers[3] = new MyTestCases.IntegerWrapper(-1);
        integers[4] = new MyTestCases.IntegerWrapper(0);
        integers[5] = new MyTestCases.IntegerWrapper(1);
        integers[6] = new MyTestCases.IntegerWrapper(2);
        integers[7] = new MyTestCases.IntegerWrapper(3);
        integers[8] = new MyTestCases.IntegerWrapper(4);
        integers[9] = new MyTestCases.IntegerWrapper(5);

        sorted = new MyTestCases.IntegerWrapper[10];
        sorted[0] = new MyTestCases.IntegerWrapper(-4);
        sorted[1] = new MyTestCases.IntegerWrapper(-3);
        sorted[2] = new MyTestCases.IntegerWrapper(-2);
        sorted[3] = new MyTestCases.IntegerWrapper(-1);
        sorted[4] = new MyTestCases.IntegerWrapper(0);
        sorted[5] = new MyTestCases.IntegerWrapper(1);
        sorted[6] = new MyTestCases.IntegerWrapper(2);
        sorted[7] = new MyTestCases.IntegerWrapper(3);
        sorted[8] = new MyTestCases.IntegerWrapper(4);
        sorted[9] = new MyTestCases.IntegerWrapper(5);

        Sorting.cocktailSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 34 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testCocktailSortSingleElement() {
        integers = new MyTestCases.IntegerWrapper[1];
        integers[0] = new MyTestCases.IntegerWrapper(0);

        sorted = new MyTestCases.IntegerWrapper[1];
        sorted[0] = new MyTestCases.IntegerWrapper(0);

        Sorting.cocktailSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() == 0);
    }

    @Test(timeout = TIMEOUT)
    public void testMergeSortException() {
        integers = new MyTestCases.IntegerWrapper[8];
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.mergeSort(integers, null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testMergeSortException2() {
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.mergeSort(null, comp);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testMergeSort1() {
        integers = new MyTestCases.IntegerWrapper[5];
        for (int i = 0; i < 5; i++) {
            integers[i] = new MyTestCases.IntegerWrapper(4 - i);
        }

        sorted = new MyTestCases.IntegerWrapper[5];
        for (int i = 0; i < 5; i++) {
            sorted[i] = new MyTestCases.IntegerWrapper(i);
        }

        Sorting.mergeSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 7 && comp.getCount() != 0);
    }
    @Test(timeout = TIMEOUT)
    public void testMergeSort2() {
        IntegerWrapper a = new IntegerWrapper(2);
        IntegerWrapper b = new IntegerWrapper(2);
        IntegerWrapper c = new IntegerWrapper(1);
        integers = new MyTestCases.IntegerWrapper[3];
        integers[0] = a;
        integers[1] = b;
        integers[2] = c;

        sorted = new MyTestCases.IntegerWrapper[3];
        sorted[0] = c;
        sorted[1] = a;
        sorted[2] = b;

        Sorting.mergeSort(integers, comp);
        assertArrayEquals(sorted, integers);
        for (int i = 0; i < integers.length; i++) {
            //Tests stability
            assertTrue(integers[i] == sorted[i]);
        }

        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 3 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testMergeSort3() {
        integers = new MyTestCases.IntegerWrapper[10];
        integers[0] = new MyTestCases.IntegerWrapper(3);
        integers[1] = new MyTestCases.IntegerWrapper(-1);
        integers[2] = new MyTestCases.IntegerWrapper(0);
        integers[3] = new MyTestCases.IntegerWrapper(4);
        integers[4] = new MyTestCases.IntegerWrapper(-3);
        integers[5] = new MyTestCases.IntegerWrapper(1);
        integers[6] = new MyTestCases.IntegerWrapper(-2);
        integers[7] = new MyTestCases.IntegerWrapper(-4);
        integers[8] = new MyTestCases.IntegerWrapper(5);
        integers[9] = new MyTestCases.IntegerWrapper(2);

        sorted = new MyTestCases.IntegerWrapper[10];
        sorted[0] = new MyTestCases.IntegerWrapper(-4);
        sorted[1] = new MyTestCases.IntegerWrapper(-3);
        sorted[2] = new MyTestCases.IntegerWrapper(-2);
        sorted[3] = new MyTestCases.IntegerWrapper(-1);
        sorted[4] = new MyTestCases.IntegerWrapper(0);
        sorted[5] = new MyTestCases.IntegerWrapper(1);
        sorted[6] = new MyTestCases.IntegerWrapper(2);
        sorted[7] = new MyTestCases.IntegerWrapper(3);
        sorted[8] = new MyTestCases.IntegerWrapper(4);
        sorted[9] = new MyTestCases.IntegerWrapper(5);

        Sorting.mergeSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 23 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testMergeSortAlreadySorted() {
        integers = new MyTestCases.IntegerWrapper[10];
        integers[0] = new MyTestCases.IntegerWrapper(-4);
        integers[1] = new MyTestCases.IntegerWrapper(-3);
        integers[2] = new MyTestCases.IntegerWrapper(-2);
        integers[3] = new MyTestCases.IntegerWrapper(-1);
        integers[4] = new MyTestCases.IntegerWrapper(0);
        integers[5] = new MyTestCases.IntegerWrapper(1);
        integers[6] = new MyTestCases.IntegerWrapper(2);
        integers[7] = new MyTestCases.IntegerWrapper(3);
        integers[8] = new MyTestCases.IntegerWrapper(4);
        integers[9] = new MyTestCases.IntegerWrapper(5);

        sorted = new MyTestCases.IntegerWrapper[10];
        sorted[0] = new MyTestCases.IntegerWrapper(-4);
        sorted[1] = new MyTestCases.IntegerWrapper(-3);
        sorted[2] = new MyTestCases.IntegerWrapper(-2);
        sorted[3] = new MyTestCases.IntegerWrapper(-1);
        sorted[4] = new MyTestCases.IntegerWrapper(0);
        sorted[5] = new MyTestCases.IntegerWrapper(1);
        sorted[6] = new MyTestCases.IntegerWrapper(2);
        sorted[7] = new MyTestCases.IntegerWrapper(3);
        sorted[8] = new MyTestCases.IntegerWrapper(4);
        sorted[9] = new MyTestCases.IntegerWrapper(5);

        Sorting.mergeSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 15 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testMergeSortSingleElement() {
        integers = new MyTestCases.IntegerWrapper[1];
        integers[0] = new MyTestCases.IntegerWrapper(0);

        sorted = new MyTestCases.IntegerWrapper[1];
        sorted[0] = new MyTestCases.IntegerWrapper(0);

        Sorting.mergeSort(integers, comp);
        assertArrayEquals(sorted, integers);
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() == 0);
    }

    @Test(timeout = TIMEOUT)
    public void testKthSelectException() {
        integers = new MyTestCases.IntegerWrapper[1];
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.kthSelect(1, null, comp, new Random());
        });
    }

    @Test(timeout = TIMEOUT)
    public void testKthSelectException2() {
        integers = new MyTestCases.IntegerWrapper[1];
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.kthSelect(1, integers, null, new Random());
        });
    }

    @Test(timeout = TIMEOUT)
    public void testKthSelectException3() {
        integers = new MyTestCases.IntegerWrapper[1];
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.kthSelect(1, integers, comp, null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testKthSelectOutOfRangeExceptions() {
        integers = new MyTestCases.IntegerWrapper[1];
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.kthSelect(0, integers, comp, new Random());
        });
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.kthSelect(2, integers, comp, new Random());
        });
    }

    @Test(timeout = TIMEOUT)
    public void testKthSelect1() {
        int rSeed = 8;
        integers = new MyTestCases.IntegerWrapper[10];
        integers[0] = new MyTestCases.IntegerWrapper(3);
        integers[1] = new MyTestCases.IntegerWrapper(-1);
        integers[2] = new MyTestCases.IntegerWrapper(0);
        integers[3] = new MyTestCases.IntegerWrapper(4);
        integers[4] = new MyTestCases.IntegerWrapper(-3);
        integers[5] = new MyTestCases.IntegerWrapper(1);
        integers[6] = new MyTestCases.IntegerWrapper(-2);
        integers[7] = new MyTestCases.IntegerWrapper(-4);
        integers[8] = new MyTestCases.IntegerWrapper(5);
        integers[9] = new MyTestCases.IntegerWrapper(2);

        assertEquals(new IntegerWrapper(-4), Sorting.kthSelect(1,
                integers, comp, new Random(rSeed)));
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 16 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(0), Sorting.kthSelect(5,
                integers, comp, new Random(rSeed)));
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 31 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(5), Sorting.kthSelect(10,
                integers, comp, new Random(rSeed)));
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 18 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testKthSelect2() {
        int rSeed = 8;
        integers = new MyTestCases.IntegerWrapper[50];
        for (int i = 0; i < 50; i++) {
            integers[i] = new MyTestCases.IntegerWrapper(49 - i);
        }

        assertEquals(new IntegerWrapper(10), Sorting.kthSelect(11,
                integers, comp, new Random(rSeed)));
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 180 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(49), Sorting.kthSelect(50,
                integers, comp, new Random(rSeed)));
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 117 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(17), Sorting.kthSelect(18,
                integers, comp, new Random(rSeed)));
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 103 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testKthSelectAlreadySorted() {
        int rSeed = 8;
        integers = new MyTestCases.IntegerWrapper[10];
        integers[0] = new MyTestCases.IntegerWrapper(-4);
        integers[1] = new MyTestCases.IntegerWrapper(-3);
        integers[2] = new MyTestCases.IntegerWrapper(-2);
        integers[3] = new MyTestCases.IntegerWrapper(-1);
        integers[4] = new MyTestCases.IntegerWrapper(0);
        integers[5] = new MyTestCases.IntegerWrapper(1);
        integers[6] = new MyTestCases.IntegerWrapper(2);
        integers[7] = new MyTestCases.IntegerWrapper(3);
        integers[8] = new MyTestCases.IntegerWrapper(4);
        integers[9] = new MyTestCases.IntegerWrapper(5);

        assertEquals(new IntegerWrapper(0), Sorting.kthSelect(5,
                integers, comp, new Random(rSeed)));
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 10 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(-2), Sorting.kthSelect(3,
                integers, comp, new Random(rSeed)));
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 16 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(5), Sorting.kthSelect(10,
                integers, comp, new Random(rSeed)));
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 18 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testKthSelectSingleElement() {
        int rSeed = 8;
        integers = new MyTestCases.IntegerWrapper[1];
        integers[0] = new MyTestCases.IntegerWrapper(0);

        assertEquals(new IntegerWrapper(0), Sorting.kthSelect(1,
                integers, comp, new Random(rSeed)));
        assertTrue("Comparisons: " + comp.getCount(),
                comp.getCount() <= 0);
    }

    @Test(timeout = TIMEOUT)
    public void testLsdRadixSortException() {
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.lsdRadixSort(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort1() {
        int[] unsortedArray = new int[]{12, -1218, 1, -5, -18, -205, -12, 5, -23451, 450};
        int[] sortedArray = new int[]{-23451, -1218, -205, -18, -12, -5, 1, 5, 12, 450};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }

    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort2() {
        int[] unsortedArray = new int[]{-18, 16, -14, 12, -10, 8, -6, 4, -2, 0,
                -22, 444, -6666, 88888, -111111, 2222222, 20000};
        int[] sortedArray = new int[]{-111111, -6666, -22, -18, -14, -10, -6, -2, 0,
                4, 8, 12, 16, 444, 20000, 88888, 2222222};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }

    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort3() {
        int[] unsortedArray = new int[100];
        for (int i = 0; i < 100; i++) {
            unsortedArray[i] = 50 - i;
        }
        int[] sortedArray = new int[100];
        for (int i = -49; i <= 50; i++) {
            sortedArray[i + 49] = i;
        }
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }

    @Test(timeout = TIMEOUT)
    public void testLsdRadixSortAlreadySorted() {
        int[] unsortedArray = new int[]{-4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
        int[] sortedArray = new int[]{-4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }

    @Test(timeout = TIMEOUT)
    public void testLsdRadixSortAlreadySorted2() {
        int[] unsortedArray = new int[]{1, 22, 333, 4444, 55555, 666666};
        int[] sortedArray = new int[]{1, 22, 333, 4444, 55555, 666666};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }

    @Test(timeout = TIMEOUT)
    public void testLsdRadixSortSingleElement() {
        int[] unsortedArray = new int[]{1};
        int[] sortedArray = new int[]{1};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }

    @Test(timeout = TIMEOUT)
    public void testHeapSortException() {
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.heapSort(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testHeapSort1() {
        int[] unsortedArray = new int[]{12, -1218, 1, -5, -18, -205, -12, 5, -23451, 450};
        List<Integer> unsortedList = new ArrayList<>();
        for (int i : unsortedArray) {
            unsortedList.add(i);
        }
        int[] sortedArray = new int[]{-23451, -1218, -205, -18, -12, -5, 1, 5, 12, 450};
        int[] actualArray = Sorting.heapSort(unsortedList);
        assertArrayEquals(sortedArray, actualArray);
    }

    @Test(timeout = TIMEOUT)
    public void testHeapSort2() {
        List<Integer> unsortedList = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            unsortedList.add(99 - i);
        }
        int[] sortedArray = new int[100];
        for (int i = 0; i < 100; i++) {
            sortedArray[i] = i;
        }
        int[] actualArray = Sorting.heapSort(unsortedList);
        assertArrayEquals(sortedArray, actualArray);
    }

    @Test(timeout = TIMEOUT)
    public void testHeapSortAlreadySorted() {
        int[] unsortedArray = new int[]{-23451, -1218, -205, -18, -12, -5, 1, 5, 12, 450};
        List<Integer> unsortedList = new ArrayList<>();
        for (int i : unsortedArray) {
            unsortedList.add(i);
        }
        int[] sortedArray = new int[]{-23451, -1218, -205, -18, -12, -5, 1, 5, 12, 450};
        int[] actualArray = Sorting.heapSort(unsortedList);
        assertArrayEquals(sortedArray, actualArray);
    }

    @Test(timeout = TIMEOUT)
    public void testHeapSortSingleElement() {
        int[] unsortedArray = new int[]{1};
        List<Integer> unsortedList = new ArrayList<>();
        for (int i : unsortedArray) {
            unsortedList.add(i);
        }
        int[] sortedArray = new int[]{1};
        int[] actualArray = Sorting.heapSort(unsortedList);
        assertArrayEquals(sortedArray, actualArray);
    }
}