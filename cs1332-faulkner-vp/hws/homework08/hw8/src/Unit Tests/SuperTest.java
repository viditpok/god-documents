import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;
import static org.junit.Assert.assertThrows;

/**
 * @author Rishi Soni
 * @version 1.0
 */
public class SuperTest {

    private static final int TIMEOUT = 200;
    private IntegerWrapper[] integers;
    private IntegerWrapper[] sortedIntegers;
    private ComparatorPlus<IntegerWrapper> comp;

    @Before
    public void setUp() {
        comp = IntegerWrapper.getComparator();
    }

    @Test(timeout = TIMEOUT)
    public void selectionSortExceptions() {
        integers = new SuperTest.IntegerWrapper[8];
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.selectionSort(null, comp);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.selectionSort(integers, null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void cocktailSortExceptions() {
        integers = new SuperTest.IntegerWrapper[8];
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.cocktailSort(null, comp);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.cocktailSort(integers, null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void mergeSortExceptions() {
        integers = new SuperTest.IntegerWrapper[8];
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.mergeSort(null, comp);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.mergeSort(integers, null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void kthSelectExceptions() {
        integers = new SuperTest.IntegerWrapper[1];
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.kthSelect(1, null, comp, new Random());
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.kthSelect(1, integers, null, new Random());
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.kthSelect(1, integers, comp, null);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.kthSelect(0, integers, comp, new Random());
        });

        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.kthSelect(2, integers, comp, new Random());
        });
    }

    @Test(timeout = TIMEOUT)
    public void lsdRadixSortExceptions() {
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.lsdRadixSort(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void heapSortExceptions() {
        assertThrows(IllegalArgumentException.class, () -> {
            Sorting.heapSort(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void selectionSort1() {
        integers = new SuperTest.IntegerWrapper[10];
        integers[0] = new SuperTest.IntegerWrapper(9);
        integers[1] = new SuperTest.IntegerWrapper(5);
        integers[2] = new SuperTest.IntegerWrapper(30);
        integers[3] = new SuperTest.IntegerWrapper(5);
        integers[4] = new SuperTest.IntegerWrapper(7);
        integers[5] = new SuperTest.IntegerWrapper(1);
        integers[6] = new SuperTest.IntegerWrapper(-3);
        integers[7] = new SuperTest.IntegerWrapper(2);
        integers[8] = new SuperTest.IntegerWrapper(50);
        integers[9] = new SuperTest.IntegerWrapper(30);

        sortedIntegers = new SuperTest.IntegerWrapper[10];
        sortedIntegers[0] = new SuperTest.IntegerWrapper(-3);
        sortedIntegers[1] = new SuperTest.IntegerWrapper(1);
        sortedIntegers[2] = new SuperTest.IntegerWrapper(2);
        sortedIntegers[3] = new SuperTest.IntegerWrapper(5);
        sortedIntegers[4] = new SuperTest.IntegerWrapper(5);
        sortedIntegers[5] = new SuperTest.IntegerWrapper(7);
        sortedIntegers[6] = new SuperTest.IntegerWrapper(9);
        sortedIntegers[7] = new SuperTest.IntegerWrapper(30);
        sortedIntegers[8] = new SuperTest.IntegerWrapper(30);
        sortedIntegers[9] = new SuperTest.IntegerWrapper(50);

        Sorting.selectionSort(integers, comp);
        assertArrayEquals(sortedIntegers, integers);
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 45 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void selectionSort2() {
        integers = new SuperTest.IntegerWrapper[2];
        integers[0] = new SuperTest.IntegerWrapper(0);
        integers[1] = new SuperTest.IntegerWrapper(-1);

        sortedIntegers = new SuperTest.IntegerWrapper[2];
        sortedIntegers[0] = new SuperTest.IntegerWrapper(-1);
        sortedIntegers[1] = new SuperTest.IntegerWrapper(0);

        Sorting.selectionSort(integers, comp);
        assertArrayEquals(sortedIntegers, integers);
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() == 1);
    }

    @Test(timeout = TIMEOUT)
    public void cocktailSort1() {
        integers = new SuperTest.IntegerWrapper[10];
        for (int i = 0; i < 10; i++) {
            integers[i] = new SuperTest.IntegerWrapper(9 - i);
        }

        sortedIntegers = new SuperTest.IntegerWrapper[10];
        for (int i = 0; i < 10; i++) {
            sortedIntegers[i] = new SuperTest.IntegerWrapper(i);
        }

        Sorting.cocktailSort(integers, comp);
        assertArrayEquals(sortedIntegers, integers);
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 45 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void cocktailSort2() {
        IntegerWrapper a = new IntegerWrapper(0);
        IntegerWrapper b = new IntegerWrapper(0);
        IntegerWrapper c = new IntegerWrapper(0);
        IntegerWrapper d = new IntegerWrapper(-1);
        integers = new SuperTest.IntegerWrapper[4];
        integers[0] = a;
        integers[1] = b;
        integers[2] = c;
        integers[3] = d;

        sortedIntegers = new SuperTest.IntegerWrapper[4];
        sortedIntegers[0] = d;
        sortedIntegers[1] = a;
        sortedIntegers[2] = b;
        sortedIntegers[3] = c;

        Sorting.cocktailSort(integers, comp);
        assertArrayEquals(sortedIntegers, integers);
        for (int i = 0; i < integers.length; i++) {
            //Checking whether stable through reference equality
            assertTrue(integers[i] == sortedIntegers[i]);
        }

        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 6 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void cocktailSort3() {
        integers = new SuperTest.IntegerWrapper[10];
        integers[0] = new SuperTest.IntegerWrapper(9);
        integers[1] = new SuperTest.IntegerWrapper(5);
        integers[2] = new SuperTest.IntegerWrapper(30);
        integers[3] = new SuperTest.IntegerWrapper(5);
        integers[4] = new SuperTest.IntegerWrapper(7);
        integers[5] = new SuperTest.IntegerWrapper(1);
        integers[6] = new SuperTest.IntegerWrapper(-3);
        integers[7] = new SuperTest.IntegerWrapper(2);
        integers[8] = new SuperTest.IntegerWrapper(50);
        integers[9] = new SuperTest.IntegerWrapper(30);

        sortedIntegers = new SuperTest.IntegerWrapper[10];
        sortedIntegers[0] = new SuperTest.IntegerWrapper(-3);
        sortedIntegers[1] = new SuperTest.IntegerWrapper(1);
        sortedIntegers[2] = new SuperTest.IntegerWrapper(2);
        sortedIntegers[3] = new SuperTest.IntegerWrapper(5);
        sortedIntegers[4] = new SuperTest.IntegerWrapper(5);
        sortedIntegers[5] = new SuperTest.IntegerWrapper(7);
        sortedIntegers[6] = new SuperTest.IntegerWrapper(9);
        sortedIntegers[7] = new SuperTest.IntegerWrapper(30);
        sortedIntegers[8] = new SuperTest.IntegerWrapper(30);
        sortedIntegers[9] = new SuperTest.IntegerWrapper(50);

        Sorting.cocktailSort(integers, comp);
        assertArrayEquals(sortedIntegers, integers);
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 34 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void mergeSort1() {
        integers = new SuperTest.IntegerWrapper[10];
        for (int i = 0; i < 10; i++) {
            integers[i] = new SuperTest.IntegerWrapper(9 - i);
        }

        sortedIntegers = new SuperTest.IntegerWrapper[10];
        for (int i = 0; i < 10; i++) {
            sortedIntegers[i] = new SuperTest.IntegerWrapper(i);
        }

        Sorting.mergeSort(integers, comp);
        assertArrayEquals(sortedIntegers, integers);
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 19 && comp.getCount() != 0);
    }
    @Test(timeout = TIMEOUT)
    public void mergeSort2() {
        IntegerWrapper a = new IntegerWrapper(0);
        IntegerWrapper b = new IntegerWrapper(0);
        IntegerWrapper c = new IntegerWrapper(0);
        IntegerWrapper d = new IntegerWrapper(-1);
        integers = new SuperTest.IntegerWrapper[4];
        integers[0] = a;
        integers[1] = b;
        integers[2] = c;
        integers[3] = d;

        sortedIntegers = new SuperTest.IntegerWrapper[4];
        sortedIntegers[0] = d;
        sortedIntegers[1] = a;
        sortedIntegers[2] = b;
        sortedIntegers[3] = c;

        Sorting.mergeSort(integers, comp);
        assertArrayEquals(sortedIntegers, integers);
        for (int i = 0; i < integers.length; i++) {
            //Checking whether stable through reference equality
            assertTrue(integers[i] == sortedIntegers[i]);
        }

        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 5 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void mergeSort3() {
        integers = new SuperTest.IntegerWrapper[10];
        integers[0] = new SuperTest.IntegerWrapper(9);
        integers[1] = new SuperTest.IntegerWrapper(5);
        integers[2] = new SuperTest.IntegerWrapper(30);
        integers[3] = new SuperTest.IntegerWrapper(5);
        integers[4] = new SuperTest.IntegerWrapper(7);
        integers[5] = new SuperTest.IntegerWrapper(1);
        integers[6] = new SuperTest.IntegerWrapper(-3);
        integers[7] = new SuperTest.IntegerWrapper(2);
        integers[8] = new SuperTest.IntegerWrapper(50);
        integers[9] = new SuperTest.IntegerWrapper(30);

        sortedIntegers = new SuperTest.IntegerWrapper[10];
        sortedIntegers[0] = new SuperTest.IntegerWrapper(-3);
        sortedIntegers[1] = new SuperTest.IntegerWrapper(1);
        sortedIntegers[2] = new SuperTest.IntegerWrapper(2);
        sortedIntegers[3] = new SuperTest.IntegerWrapper(5);
        sortedIntegers[4] = new SuperTest.IntegerWrapper(5);
        sortedIntegers[5] = new SuperTest.IntegerWrapper(7);
        sortedIntegers[6] = new SuperTest.IntegerWrapper(9);
        sortedIntegers[7] = new SuperTest.IntegerWrapper(30);
        sortedIntegers[8] = new SuperTest.IntegerWrapper(30);
        sortedIntegers[9] = new SuperTest.IntegerWrapper(50);

        Sorting.mergeSort(integers, comp);
        assertArrayEquals(sortedIntegers, integers);
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 21 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void kthSelect1() {
        int randomSeed = 15;
        integers = new SuperTest.IntegerWrapper[10];
        integers[0] = new SuperTest.IntegerWrapper(9);
        integers[1] = new SuperTest.IntegerWrapper(5);
        integers[2] = new SuperTest.IntegerWrapper(30);
        integers[3] = new SuperTest.IntegerWrapper(5);
        integers[4] = new SuperTest.IntegerWrapper(7);
        integers[5] = new SuperTest.IntegerWrapper(1);
        integers[6] = new SuperTest.IntegerWrapper(-3);
        integers[7] = new SuperTest.IntegerWrapper(2);
        integers[8] = new SuperTest.IntegerWrapper(50);
        integers[9] = new SuperTest.IntegerWrapper(-30);

        assertEquals(new IntegerWrapper(-30), Sorting.kthSelect(1,
                integers, comp, new Random(randomSeed)));
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 16 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(5), Sorting.kthSelect(5,
                integers, comp, new Random(randomSeed)));
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 26 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(50), Sorting.kthSelect(10,
                integers, comp, new Random(randomSeed)));
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 19 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void kthSelect2() {
        int randomSeed = 15;
        integers = new SuperTest.IntegerWrapper[100];
        for (int i = 0; i < 100; i++) {
            integers[i] = new SuperTest.IntegerWrapper(99 - i);
        }

        assertEquals(new IntegerWrapper(20), Sorting.kthSelect(21,
                integers, comp, new Random(randomSeed)));
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 317 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(99), Sorting.kthSelect(100,
                integers, comp, new Random(randomSeed)));
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 173 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(35), Sorting.kthSelect(36,
                integers, comp, new Random(randomSeed)));
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 561 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void kthSelect3() {
        int randomSeed = 15;
        integers = new SuperTest.IntegerWrapper[20];
        for (int i = 0; i < 20; i++) {
            integers[i] = new SuperTest.IntegerWrapper(0);
        }

        assertEquals(new IntegerWrapper(0), Sorting.kthSelect(10,
                integers, comp, new Random(randomSeed)));
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 154 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(0), Sorting.kthSelect(8,
                integers, comp, new Random(randomSeed)));
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 169 && comp.getCount() != 0);
        comp.count = 0;

        assertEquals(new IntegerWrapper(0), Sorting.kthSelect(20,
                integers, comp, new Random(randomSeed)));
        assertTrue("Number of comparisons: " + comp.getCount(),
                comp.getCount() <= 19 && comp.getCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void lsdRadixSort1() {
        int[] unsortedArray = new int[]{-18, 2223, -1, 5, 18, 300, 27, 0, 123456, -555};
        int[] sortedArray = new int[]{-555, -18, -1, 0, 5, 18, 27, 300, 2223, 123456};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }

    @Test(timeout = TIMEOUT)
    public void lsdRadixSort2() {
        int[] unsortedArray = new int[]{-9, 8, -7, 6, -5, 4, -3, 2, -1, 0, -11, 222, -3333, 44444, -555555, 6666666, 10000};
        int[] sortedArray = new int[]{-555555, -3333, -11, -9, -7, -5, -3, -1, 0, 2, 4, 6, 8, 222, 10000, 44444, 6666666};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }

    @Test(timeout = TIMEOUT)
    public void lsdRadixSort3() {
        int[] unsortedArray = new int[1000];
        for (int i = 0; i < 1000; i++) {
            unsortedArray[i] = 500 - i;
        }
        int[] sortedArray = new int[1000];
        for (int i = -499; i <= 500; i++) {
            sortedArray[i + 499] = i;
        }
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }

    @Test(timeout = TIMEOUT)
    public void heapSort1() {
        int[] unsortedArray = new int[] {3, -4, -18, 0, 5, 43, 10, 7, 5, 10, 18};
        List<Integer> unsortedList = new ArrayList<>();
        for (int i : unsortedArray) {
            unsortedList.add(i);
        }
        int[] sortedArray = new int[] {-18, -4, 0, 3, 5, 5, 7, 10, 10, 18, 43};
        int[] actualArray = Sorting.heapSort(unsortedList);
        assertArrayEquals(sortedArray, actualArray);
    }

    @Test(timeout = TIMEOUT)
    public void heapSort2() {
        int[] unsortedArray = new int[] {-17, 18, 3, 4, 3, 8, 10, 24, 16, -1, 50000};
        List<Integer> unsortedList = new ArrayList<>();
        for (int i : unsortedArray) {
            unsortedList.add(i);
        }
        int[] sortedArray = new int[] {-17, -1, 3, 3, 4, 8, 10, 16, 18, 24, 50000};
        int[] actualArray = Sorting.heapSort(unsortedList);
        assertArrayEquals(sortedArray, actualArray);
    }

    @Test(timeout = TIMEOUT)
    public void heapSort3() {
        List<Integer> unsortedList = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            unsortedList.add(999 - i);
        }
        int[] sortedArray = new int[1000];
        for (int i = 0; i < 1000; i++) {
            sortedArray[i] = i;
        }
        int[] actualArray = Sorting.heapSort(unsortedList);
        assertArrayEquals(sortedArray, actualArray);
    }

    /**
     * Class for testing proper sorting.
     */
    private static class IntegerWrapper {
        private Integer value;

        /**
         * Create an IntegerWrapper.
         *
         * @param value integer value
         */
        public IntegerWrapper(Integer value) {
            this.value = value;
        }

        /**
         * Get the value
         *
         * @return value of the integer
         */
        public Integer getValue() {
            return value;
        }

        /**
         * Set the value of the IntegerWrapper.
         *
         * @param value the new value
         */
        public void setValue(Integer value) {
            this.value = value;
        }

        /**
         * Create a comparator that compares the wrapped values.
         *
         * @return comparator that compares the wrapped values
         */
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

    /**
     * Inner class that allows counting how many comparisons were made.
     */
    private abstract static class ComparatorPlus<T> implements Comparator<T> {
        private int count;

        /**
         * Get the number of comparisons made.
         *
         * @return number of comparisons made
         */
        public int getCount() {
            return count;
        }

        /**
         * Increment the number of comparisons made by one. Call this method in
         * your compare() implementation.
         */
        public void incrementCount() {
            count++;
        }
    }
}
