import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

/**
 * Sorting Algorithm JUnit Tests
 * 3/16/2023
 * @author Lucian Tash
 * @version 1.0
 */
public class LucianTests {

    private static final int LARGE_TEST_COUNT = 100;  // CHANGING THIS VALUE WILL MESS WITH COMPARISON COUNT CHECKS
    private static final int SMALL_TEST_COUNT = 10; // CHANGING THIS VALUE WILL MESS WITH COMPARISON COUNT CHECKS
    private static final int TIMEOUT = 200;
    private IntegerWrapper[] array;
    private IntegerWrapper[] original;
    private IntegerWrapper[] expected;
    private ComparatorPlus<IntegerWrapper> comp;

    private static void print(IntegerWrapper[] original, IntegerWrapper[] array, IntegerWrapper[] expected) {
        System.out.println();
        print("Original", original);
        print("Your Array", array);
        print("Expected", expected);
    }

    private static String getArrAsString(IntegerWrapper[] arr) {
        String str = "";
        for (IntegerWrapper i : arr) { str += (i.getValue() + " "); }
        return str;
    }

    private static void print(int[] original, int[] array, int[] expected) {
        System.out.print("\n\nOriginal:\t");
        print(original);
        System.out.print("\nYour Array:\t");
        print(array);
        System.out.print("\nExpected:\t");
        print(expected);
    }

    private static void print(int[] arr) {
        for (int i : arr) { System.out.print(i + " "); }
    }

    private static void print(String label, IntegerWrapper[] arr) {
        System.out.println(label + ":\t" + getArrAsString(arr));
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testSelectionSortNullArray() {
        Sorting.selectionSort(null, comp);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testSelectionSortNullComparator() {
        Sorting.selectionSort(array, null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testCocktailSortNullArray() {
        Sorting.cocktailSort(null, comp);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testCocktailSortNullComparator() {
        Sorting.cocktailSort(array, null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testMergeSortNullArray() {
        Sorting.mergeSort(null, comp);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testMergeSortNullComparator() {
        Sorting.mergeSort(array, null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testKthSelectNullArray() {
        comp = IntegerWrapper.getComparator();
        Sorting.kthSelect(1, null, comp, new Random());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testKthSelectNullComparator() {
        array = new IntegerWrapper[5];
        Sorting.kthSelect(1, array, null, new Random());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testKthSelectAboveBounds() {
        array = new IntegerWrapper[5];
        comp = IntegerWrapper.getComparator();
        Sorting.kthSelect(6, array, comp, new Random());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testKthSelectBelowBounds() {
        array = new IntegerWrapper[5];
        comp = IntegerWrapper.getComparator();
        Sorting.kthSelect(0, array, comp, new Random());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testRadixSortNullArray() {
        Sorting.lsdRadixSort(null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testHeapSortNullArray() {
        Sorting.heapSort(null);
    }

    @Test(timeout = TIMEOUT)
    public void testSelectionSort() {
        // ALREADY SORTED ARRAY - List of ten integers from 0-9, in ascending order
        array = new IntegerWrapper[SMALL_TEST_COUNT];
        original = new IntegerWrapper[SMALL_TEST_COUNT];
        expected = new IntegerWrapper[SMALL_TEST_COUNT];
        comp = IntegerWrapper.getComparator();
        for (int i = 0; i < SMALL_TEST_COUNT; i++) {
            array[i] = original[i] = expected[i] = new IntegerWrapper(i);
        }
        Sorting.selectionSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
        assertTrue("Number of comparisons: " + comp.getCount(),comp.getCount() <= 45);

        // REVERSE SORTED ARRAY
        comp = IntegerWrapper.getComparator();
        for (int i = 0; i < SMALL_TEST_COUNT; i++) {
            array[i] = original[i] = new IntegerWrapper(SMALL_TEST_COUNT - i - 1);
        }
        Sorting.selectionSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
        assertTrue("Number of comparisons: " + comp.getCount(),comp.getCount() <= 45);

        // SHUFFLED ARRAY
        comp = IntegerWrapper.getComparator();
        array[0] = original[0] = new IntegerWrapper(1);
        array[1] = original[1] = new IntegerWrapper(4);
        array[2] = original[2] = new IntegerWrapper(2);
        array[3] = original[3] = new IntegerWrapper(5);
        array[4] = original[4] = new IntegerWrapper(6);
        array[5] = original[5] = new IntegerWrapper(9);
        array[6] = original[6] = new IntegerWrapper(7);
        array[7] = original[7] = new IntegerWrapper(8);
        array[8] = original[8] = new IntegerWrapper(3);
        array[9] = original[9] = new IntegerWrapper(0);
        Sorting.selectionSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
        assertTrue("Number of comparisons: " + comp.getCount(),comp.getCount() <= 45);

        // LARGE RANDOMIZED TEST - Randomly shuffled list with 100 ints
        comp = IntegerWrapper.getComparator();
        List<IntegerWrapper> arr1 = new ArrayList<>();
        List<IntegerWrapper> arr2 = new ArrayList<>();
        for (int i = 0; i < LARGE_TEST_COUNT; i++) {
            arr1.add(new IntegerWrapper(i));
            arr2.add(new IntegerWrapper(i));
        }
        Collections.shuffle(arr1); // makes arr1 have a random unsorted order
        array = arr1.toArray(new IntegerWrapper[LARGE_TEST_COUNT]);
        original = arr1.toArray(new IntegerWrapper[LARGE_TEST_COUNT]);
        expected = arr2.toArray(new IntegerWrapper[LARGE_TEST_COUNT]);
        Sorting.selectionSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
    }

    @Test(timeout = TIMEOUT)
    public void testCocktailSort() {
        // ALREADY SORTED ARRAY - List of ten integers from 0-9, in ascending order
        array = new IntegerWrapper[SMALL_TEST_COUNT];
        original = new IntegerWrapper[SMALL_TEST_COUNT];
        expected = new IntegerWrapper[SMALL_TEST_COUNT];
        comp = IntegerWrapper.getComparator();
        for (int i = 0; i < SMALL_TEST_COUNT; i++) {
            array[i] = original[i] = expected[i] = new IntegerWrapper(i);
        }
        Sorting.cocktailSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
        assertTrue("Number of comparisons: " + comp.getCount(),comp.getCount() <= 9);

        // REVERSE SORTED ARRAY
        comp = IntegerWrapper.getComparator();
        for (int i = 0; i < SMALL_TEST_COUNT; i++) {
            array[i] = original[i] = new IntegerWrapper(SMALL_TEST_COUNT - i - 1);
        }
        Sorting.cocktailSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
        assertTrue("Number of comparisons: " + comp.getCount(),comp.getCount() <= 45);

        // SHUFFLED ARRAY
        comp = IntegerWrapper.getComparator();
        array[0] = original[0] = new IntegerWrapper(1);
        array[1] = original[1] = new IntegerWrapper(4);
        array[2] = original[2] = new IntegerWrapper(2);
        array[3] = original[3] = new IntegerWrapper(5);
        array[4] = original[4] = new IntegerWrapper(6);
        array[5] = original[5] = new IntegerWrapper(9);
        array[6] = original[6] = new IntegerWrapper(7);
        array[7] = original[7] = new IntegerWrapper(8);
        array[8] = original[8] = new IntegerWrapper(3);
        array[9] = original[9] = new IntegerWrapper(0);
        Sorting.cocktailSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
        assertTrue("Number of comparisons: " + comp.getCount(),comp.getCount() <= 45);

        // LARGE RANDOMIZED TEST - Randomly shuffled list with 100 ints
        comp = IntegerWrapper.getComparator();
        List<IntegerWrapper> arr1 = new ArrayList<>();
        List<IntegerWrapper> arr2 = new ArrayList<>();
        for (int i = 0; i < LARGE_TEST_COUNT; i++) {
            arr1.add(new IntegerWrapper(i));
            arr2.add(new IntegerWrapper(i));
        }
        Collections.shuffle(arr1); // makes arr1 have a random unsorted order
        array = arr1.toArray(new IntegerWrapper[LARGE_TEST_COUNT]);
        original = arr1.toArray(new IntegerWrapper[LARGE_TEST_COUNT]);
        expected = arr2.toArray(new IntegerWrapper[LARGE_TEST_COUNT]);
        Sorting.cocktailSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
    }

    @Test(timeout = TIMEOUT)
    public void testMergeSort() {
        // ALREADY SORTED ARRAY - List of ten integers from 0-9, in ascending order
        array = new IntegerWrapper[SMALL_TEST_COUNT];
        original = new IntegerWrapper[SMALL_TEST_COUNT];
        expected = new IntegerWrapper[SMALL_TEST_COUNT];
        comp = IntegerWrapper.getComparator();
        for (int i = 0; i < SMALL_TEST_COUNT; i++) {
            array[i] = original[i] = expected[i] = new IntegerWrapper(i);
        }
        Sorting.mergeSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
        assertTrue("Number of comparisons: " + comp.getCount(),comp.getCount() <= 15);

        // REVERSE SORTED ARRAY
        comp = IntegerWrapper.getComparator();
        for (int i = 0; i < SMALL_TEST_COUNT; i++) {
            array[i] = original[i] = new IntegerWrapper(SMALL_TEST_COUNT - i - 1);
        }
        Sorting.mergeSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
        assertTrue("Number of comparisons: " + comp.getCount(),comp.getCount() <= 19);

        // SHUFFLED ARRAY (odd number of elements)
        comp = IntegerWrapper.getComparator();
        array = new IntegerWrapper[11];
        original = new IntegerWrapper[11];
        expected = new IntegerWrapper[11];
        array[0] = original[0] = new IntegerWrapper(1);
        array[1] = original[1] = new IntegerWrapper(4);
        array[2] = original[2] = new IntegerWrapper(2);
        array[3] = original[3] = new IntegerWrapper(5);
        array[4] = original[4] = new IntegerWrapper(6);
        array[5] = original[5] = new IntegerWrapper(9);
        array[6] = original[6] = new IntegerWrapper(7);
        array[7] = original[7] = new IntegerWrapper(8);
        array[8] = original[8] = new IntegerWrapper(3);
        array[9] = original[9] = new IntegerWrapper(0);
        array[10] = original[10] = new IntegerWrapper(10);
        for (int i = 0; i < 11; i++) { expected[i] = new IntegerWrapper(i); }
        Sorting.mergeSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
        assertTrue("Number of comparisons: " + comp.getCount(),comp.getCount() <= 24);

        // LARGE RANDOMIZED TEST - Randomly shuffled list with 100 ints
        comp = IntegerWrapper.getComparator();
        List<IntegerWrapper> arr1 = new ArrayList<>();
        List<IntegerWrapper> arr2 = new ArrayList<>();
        for (int i = 0; i < LARGE_TEST_COUNT; i++) {
            arr1.add(new IntegerWrapper(i));
            arr2.add(new IntegerWrapper(i));
        }
        Collections.shuffle(arr1); // makes arr1 have a random unsorted order
        array = arr1.toArray(new IntegerWrapper[LARGE_TEST_COUNT]);
        original = arr1.toArray(new IntegerWrapper[LARGE_TEST_COUNT]);
        expected = arr2.toArray(new IntegerWrapper[LARGE_TEST_COUNT]);
        Sorting.mergeSort(array, comp);
        print(original, array, expected);
        assertArrayEquals(expected, array);
    }

    @Test(timeout = TIMEOUT)
    public void testKthSelect() {
        // SHUFFLED ARRAY
        comp = IntegerWrapper.getComparator();
        array = new IntegerWrapper[SMALL_TEST_COUNT];
        array[0] = new IntegerWrapper(0);
        array[1] = new IntegerWrapper(4);
        array[2] = new IntegerWrapper(2);
        array[3] = new IntegerWrapper(5);
        array[4] = new IntegerWrapper(6);
        array[5] = new IntegerWrapper(9);
        array[6] = new IntegerWrapper(7);
        array[7] = new IntegerWrapper(8);
        array[8] = new IntegerWrapper(3);
        array[9] = new IntegerWrapper(1);
        print("Before", array);
        assertEquals(new IntegerWrapper(2), Sorting.kthSelect(3, array, comp, new Random(16)));
        print("After", array);
        // Extra tests
        assertEquals(new IntegerWrapper(4), Sorting.kthSelect(5, array, comp, new Random(16)));
        assertEquals(new IntegerWrapper(8), Sorting.kthSelect(9, array, comp, new Random(16)));
        assertEquals(new IntegerWrapper(9), Sorting.kthSelect(10, array, comp, new Random(16)));

        // SHUFFLED ARRAY 2
        comp = IntegerWrapper.getComparator();
        array = new IntegerWrapper[SMALL_TEST_COUNT];
        array[0] = new IntegerWrapper(12);
        array[1] = new IntegerWrapper(-5);
        array[2] = new IntegerWrapper(0);
        array[3] = new IntegerWrapper(31);
        array[4] = new IntegerWrapper(16);
        array[5] = new IntegerWrapper(25);
        array[6] = new IntegerWrapper(-10);
        array[7] = new IntegerWrapper(-12);
        array[8] = new IntegerWrapper(1);
        array[9] = new IntegerWrapper(100);
        System.out.println();
        print("Before", array);
        assertEquals(new IntegerWrapper(100), Sorting.kthSelect(10, array, comp, new Random(5)));
        print("After", array);
        // Extra tests
        assertEquals(new IntegerWrapper(-10), Sorting.kthSelect(2, array, comp, new Random(5)));
        assertEquals(new IntegerWrapper(-5), Sorting.kthSelect(3, array, comp, new Random(5)));
        assertEquals(new IntegerWrapper(0), Sorting.kthSelect(4, array, comp, new Random(5)));
        assertEquals(new IntegerWrapper(1), Sorting.kthSelect(5, array, comp, new Random(5)));
        assertEquals(new IntegerWrapper(12), Sorting.kthSelect(6, array, comp, new Random(5)));
        assertEquals(new IntegerWrapper(16), Sorting.kthSelect(7, array, comp, new Random(5)));
        assertEquals(new IntegerWrapper(25), Sorting.kthSelect(8, array, comp, new Random(5)));
        assertEquals(new IntegerWrapper(31), Sorting.kthSelect(9, array, comp, new Random(5)));
        assertEquals(new IntegerWrapper(-12), Sorting.kthSelect(1, array, comp, new Random(5)));

        // LARGE RANDOMIZED TEST - Randomly shuffled list with 100 ints
        comp = IntegerWrapper.getComparator();
        List<IntegerWrapper> arr1 = new ArrayList<>();
        for (int i = 0; i < LARGE_TEST_COUNT; i++) {
            arr1.add(new IntegerWrapper(i));
        }
        Collections.shuffle(arr1); // makes arr1 have a random unsorted order
        array = arr1.toArray(new IntegerWrapper[LARGE_TEST_COUNT]);
        System.out.println();
        print("Before", array);
        // Find 1st-100th smallest values (all of them!)
        for (int i = 1; i <= LARGE_TEST_COUNT; i++) {
            assertEquals(new IntegerWrapper(i - 1), Sorting.kthSelect(i, array, comp, new Random(16)));
        }
        print("After All", array);
    }

    @Test(timeout = TIMEOUT)
    public void testRadixSort() {
        // ALREADY SORTED ARRAY - List of 100 integers from 0 to 99, in ascending order
        int[] array = new int[LARGE_TEST_COUNT];
        int[] original = new int[LARGE_TEST_COUNT];
        int[] expected = new int[LARGE_TEST_COUNT];
        for (int i = 0; i < LARGE_TEST_COUNT; i++) {
            array[i] = original[i] = expected[i] = i;
        }
        Sorting.lsdRadixSort(array);
        print(original, array, expected);
        assertArrayEquals(expected, array);

        // REVERSE SORTED ARRAY - Note that negative numbers should come before positive
        for (int i = 0; i < LARGE_TEST_COUNT; i++) {
            array[i] = original[i] = LARGE_TEST_COUNT - i - 51;
            expected[i] = i - 50;
        }
        Sorting.lsdRadixSort(array);
        print(original, array, expected);
        assertArrayEquals(expected, array);

        // SHUFFLED ARRAY - Note that negative numbers should come before positive
        array = new int[SMALL_TEST_COUNT];
        original = new int[SMALL_TEST_COUNT];
        expected = new int[SMALL_TEST_COUNT];
        array[0] = original[0] = 11;
        array[1] = original[1] = 44;
        array[2] = original[3] = 0;
        array[3] = original[3] = -52;
        array[4] = original[4] = 6;
        array[5] = original[5] = 39;
        array[6] = original[6] = -78;
        array[7] = original[7] = 2335;
        array[8] = original[8] = 23;
        array[9] = original[9] = 190;
        expected[0] = -78;
        expected[1] = -52;
        expected[2] = 0;
        expected[3] = 6;
        expected[4] = 11;
        expected[5] = 23;
        expected[6] = 39;
        expected[7] = 44;
        expected[8] = 190;
        expected[9] = 2335;
        Sorting.lsdRadixSort(array);
        print(original, array, expected);
        assertArrayEquals(expected, array);
    }

    @Test(timeout = TIMEOUT)
    public void testHeapSort() {
        // ALREADY SORTED ARRAY - List of ten integers from 0-9, in ascending order
        int[] array = new int[SMALL_TEST_COUNT];
        int[] original = new int[SMALL_TEST_COUNT];
        int[] expected = new int[SMALL_TEST_COUNT];
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < SMALL_TEST_COUNT; i++) {
            array[i] = original[i] = expected[i] = i;
            list.add(i);
        }
        array = Sorting.heapSort(list);
        print(original, array, expected);
        assertArrayEquals(expected, array);

        // REVERSE SORTED ARRAY
        list = new ArrayList<>();
        for (int i = 0; i < SMALL_TEST_COUNT; i++) {
            array[i] = original[i] = SMALL_TEST_COUNT - i - 1;
            list.add(SMALL_TEST_COUNT - i - 1);
        }
        array = Sorting.heapSort(list);
        print(original, array, expected);
        assertArrayEquals(expected, array);

        // SHUFFLED ARRAY
        list = new ArrayList<>();
        array[0] = original[0] = 1;
        array[1] = original[1] = 4;
        array[2] = original[2] = 2;
        array[3] = original[3] = 5;
        array[4] = original[4] = 6;
        array[5] = original[5] = 9;
        array[6] = original[6] = 7;
        array[7] = original[7] = 8;
        array[8] = original[8] = 3;
        array[9] = original[9] = 0;
        list.add(1);
        list.add(4);
        list.add(2);
        list.add(5);
        list.add(6);
        list.add(9);
        list.add(7);
        list.add(8);
        list.add(3);
        list.add(0);
        array = Sorting.heapSort(list);
        print(original, array, expected);
        assertArrayEquals(expected, array);
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
                public int compare(LucianTests.IntegerWrapper int1,
                                   LucianTests.IntegerWrapper int2) {
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
