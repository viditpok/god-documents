import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.List;
import java.util.LinkedList;

/**
 * Your implementation of various sorting algorithms.
 *
 * @author Vidit Pokharna
 * @version 1.0
 * @userid vpokharna3
 * @GTID 903772087
 *
 * Collaborators: LIST ALL COLLABORATORS YOU WORKED WITH HERE
 *
 * Resources: LIST ALL NON-COURSE RESOURCES YOU CONSULTED HERE
 */
public class Sorting {

    /**
     * Implement selection sort.
     *
     * It should be:
     * in-place
     * unstable
     * not adaptive
     *
     * Have a worst case running time of:
     * O(n^2)
     *
     * And a best case running time of:
     * O(n^2)
     *
     * @param <T>        data type to sort
     * @param arr        the array that must be sorted after the method runs
     * @param comparator the Comparator used to compare the data in arr
     * @throws java.lang.IllegalArgumentException if the array or comparator is
     *                                            null
     */
    public static <T> void selectionSort(T[] arr, Comparator<T> comparator) {
        if (arr == null || comparator == null) {
            throw new IllegalArgumentException("The array or comparator is null");
        }
        int length = arr.length;

        for (int a = 0; a < length - 1; a++) {
            int min = a;
            for (int b = a + 1; b < length; b++) {
                if (comparator.compare(arr[b], arr[min]) < 0) {
                    min = b;
                }
            }
            swap(min, a, arr);
        }
    }

    /**
     * Helper method to swap two elements
     * @param <T> the type of data being swapped
     * @param arr the array from which two elements will be swapped
     * @param index1 the index of the first element to swap
     * @param index2 the index of the second element to swap
     */
    private static <T> void swap(int index1, int index2, T[] arr) {
        T temp = arr[index1];
        arr[index1] = arr[index2];
        arr[index2] = temp;
    }

    /**
     * Implement cocktail sort.
     *
     * It should be:
     * in-place
     * stable
     * adaptive
     *
     * Have a worst case running time of:
     * O(n^2)
     *
     * And a best case running time of:
     * O(n)
     *
     * NOTE: See pdf for last swapped optimization for cocktail sort. You
     * MUST implement cocktail sort with this optimization
     *
     * @param <T>        data type to sort
     * @param arr        the array that must be sorted after the method runs
     * @param comparator the Comparator used to compare the data in arr
     * @throws java.lang.IllegalArgumentException if the array or comparator is
     *                                            null
     */
    public static <T> void cocktailSort(T[] arr, Comparator<T> comparator) {
        if (arr == null || comparator == null) {
            throw new IllegalArgumentException("The array or comparator is null");
        }
        int start = 0;
        int end = arr.length - 1;

        while (start < end) {
            int swapped = start;
            for (int a = start; a < end; a++) {
                if (comparator.compare(arr[a], arr[a + 1]) > 0) {
                    swap(a, a + 1, arr);
                    swapped = a;
                }
            }
            end = swapped;
            for (int b = end; b > start; b--) {
                if (comparator.compare(arr[b], arr[b - 1]) < 0) {
                    swap(b, b - 1, arr);
                    swapped = b;
                }
            }
            start = swapped;
        }
    }

    /**
     * Implement merge sort.
     *
     * It should be:
     * out-of-place
     * stable
     * not adaptive
     *
     * Have a worst case running time of:
     * O(n log n)
     *
     * And a best case running time of:
     * O(n log n)
     *
     * You can create more arrays to run merge sort, but at the end, everything
     * should be merged back into the original T[] which was passed in.
     *
     * When splitting the array, if there is an odd number of elements, put the
     * extra data on the right side.
     *
     * Hint: If two data are equal when merging, think about which subarray
     * you should pull from first
     *
     * @param <T>        data type to sort
     * @param arr        the array to be sorted
     * @param comparator the Comparator used to compare the data in arr
     * @throws java.lang.IllegalArgumentException if the array or comparator is
     *                                            null
     */
    public static <T> void mergeSort(T[] arr, Comparator<T> comparator) {
        if (arr == null || comparator == null) {
            throw new IllegalArgumentException("The array or comparator is null");
        }

        int length = arr.length;

        if (length <= 1) {
            return;
        }

        int middleIndex = length / 2;
        T[] leftArr = (T[]) new Object[middleIndex];
        T[] rightArr = (T[]) new Object[length - middleIndex];
        int leftLength = leftArr.length;
        int rightLength = rightArr.length;


        for (int a = 0; a < middleIndex; a++) {
            leftArr[a] = arr[a];
        }

        for (int a = middleIndex; a < length; a++) {
            rightArr[a - middleIndex] = arr[a];
        }

        mergeSort(leftArr, comparator);
        mergeSort(rightArr, comparator);

        int currIndex = 0;
        int leftIndex = 0;
        int rightIndex = 0;

        while (leftIndex < leftLength && rightIndex < rightLength) {
            if (comparator.compare(leftArr[leftIndex], rightArr[rightIndex]) <= 0) {
                arr[currIndex] = leftArr[leftIndex];
                leftIndex++;
            } else {
                arr[currIndex] = rightArr[rightIndex];
                rightIndex++;
            }
            currIndex++;
        }

        while (leftIndex < leftLength) {
            arr[currIndex] = leftArr[leftIndex];
            currIndex++;
            leftIndex++;
        }

        while (rightIndex < rightLength) {
            arr[currIndex] = rightArr[rightIndex];
            currIndex++;
            rightIndex++;
        }
    }

    /**
     * Implement kth select.
     *
     * Use the provided random object to select your pivots. For example if you
     * need a pivot between a (inclusive) and b (exclusive) where b > a, use
     * the following code:
     *
     * int pivotIndex = rand.nextInt(b - a) + a;
     *
     * If your recursion uses an inclusive b instead of an exclusive one,
     * the formula changes by adding 1 to the nextInt() call:
     *
     * int pivotIndex = rand.nextInt(b - a + 1) + a;
     *
     * It should be:
     * in-place
     *
     * Have a worst case running time of:
     * O(n^2)
     *
     * And a best case running time of:
     * O(n)
     *
     * You may assume that the array doesn't contain any null elements.
     *
     * Make sure you code the algorithm as you have been taught it in class.
     * There are several versions of this algorithm and you may not get full
     * credit if you do not implement the one we have taught you!
     *
     * @param <T>        data type to sort
     * @param k          the index to retrieve data from + 1 (due to
     *                   0-indexing) if the array was sorted; the 'k' in "kth
     *                   select"; e.g. if k == 1, return the smallest element
     *                   in the array
     * @param arr        the array that should be modified after the method
     *                   is finished executing as needed
     * @param comparator the Comparator used to compare the data in arr
     * @param rand       the Random object used to select pivots
     * @return the kth smallest element
     * @throws java.lang.IllegalArgumentException if the array or comparator
     *                                            or rand is null or k is not
     *                                            in the range of 1 to arr
     *                                            .length
     */
    public static <T> T kthSelect(int k, T[] arr, Comparator<T> comparator,
                                  Random rand) {
        if (arr == null || comparator == null || rand == null) {
            throw new IllegalArgumentException("The array, comparator, or random object is null");
        } else if (k < 1 || k > arr.length) {
            throw new IllegalArgumentException("The value of k is invalid");
        }
        int left = 0;
        int right = arr.length - 1;

        while (true) {
            int pivotIndex = rand.nextInt(right - left + 1) + left;
            int j = quickSelect(arr, left, right, pivotIndex, comparator);
            if (k - 1 == j) {
                return arr[k - 1];
            } else if (k - 1 < j) {
                right = j - 1;
            } else {
                left = j + 1;
            }
        }
    }

    /**
     * Helper method to use the quickselect algorithm for kselect
     * @param <T> the type of data being swapped
     * @param arr the array which the quickselect algorithm will utilize
     * @param left the index of the leftmost element in the array
     * @param right the index of the rightmost element in the array
     * @param pivotIndex the index of pivot found in kselect
     * @param comparator the object used to compare to elements
     * @return the value of int j, which will be compared to k
     */
    private static <T> int quickSelect(T[] arr, int left, int right, int pivotIndex, Comparator<T> comparator) {
        T pivotValue = arr[pivotIndex];
        int i = left + 1;
        int j = right;
        swap(left, pivotIndex, arr);
        while (i <= j) {
            while (i <= j && comparator.compare(arr[i], pivotValue) <= 0) {
                i++;
            }
            while (i <= j && comparator.compare(arr[j], pivotValue) >= 0) {
                j--;
            }
            if (i <= j) {
                swap(i, j, arr);
                i++;
                j--;
            }
        }
        swap(left, j, arr);
        return j;
    }

    /**
     * Implement LSD (least significant digit) radix sort.
     *
     * Make sure you code the algorithm as you have been taught it in class.
     * There are several versions of this algorithm and you may not get full
     * credit if you do not implement the one we have taught you!
     *
     * Remember you CANNOT convert the ints to strings at any point in your
     * code! Doing so may result in a 0 for the implementation.
     *
     * It should be:
     * out-of-place
     * stable
     * not adaptive
     *
     * Have a worst case running time of:
     * O(kn)
     *
     * And a best case running time of:
     * O(kn)
     *
     * You are allowed to make an initial O(n) passthrough of the array to
     * determine the number of iterations you need. The number of iterations
     * can be determined using the number with the largest magnitude.
     *
     * At no point should you find yourself needing a way to exponentiate a
     * number; any such method would be non-O(1). Think about how how you can
     * get each power of BASE naturally and efficiently as the algorithm
     * progresses through each digit.
     *
     * Refer to the PDF for more information on LSD Radix Sort.
     *
     * You may use ArrayList or LinkedList if you wish, but it may only be
     * used inside radix sort and any radix sort helpers. Do NOT use these
     * classes with other sorts. However, be sure the List implementation you
     * choose allows for stability while being as efficient as possible.
     *
     * Do NOT use anything from the Math class except Math.abs().
     *
     * @param arr the array to be sorted
     * @throws java.lang.IllegalArgumentException if the array is null
     */
    public static void lsdRadixSort(int[] arr) {
        if (arr == null) {
            throw new IllegalArgumentException("The array is null");
        }

        LinkedList<Integer>[] buckets = new LinkedList[19];
        for (int a = 0; a < buckets.length; a++) {
            buckets[a] = new LinkedList<>();
        }

        int maxNumber = 0;
        for (int a : arr) {
            if (Math.abs(a) > maxNumber) {
                maxNumber = Math.abs(a);
            }
        }
        int iterations = 0;
        while (maxNumber > 0) {
            iterations++;
            maxNumber /= 10;
        }

        int divide = 1;

        for (int a = 0; a < iterations; a++) {
            for (int b : arr) {
                int ithDigit = ((int) ((Math.abs((long) b)) / divide)) % 10;
                if (b < 0) {
                    ithDigit *= -1;
                }
                buckets[9 + ithDigit].add(b);
            }
            int index = 0;
            for (LinkedList<Integer> bucket : buckets) {
                while (!bucket.isEmpty()) {
                    arr[index] = bucket.removeFirst();
                    index++;
                }
            }
            divide *= 10;
        }
    }

    /**
     * Implement heap sort.
     *
     * It should be:
     * out-of-place
     * unstable
     * not adaptive
     *
     * Have a worst case running time of:
     * O(n log n)
     *
     * And a best case running time of:
     * O(n log n)
     *
     * Use java.util.PriorityQueue as the heap. Note that in this
     * PriorityQueue implementation, elements are removed from smallest
     * element to largest element.
     *
     * Initialize the PriorityQueue using its build heap constructor (look at
     * the different constructors of java.util.PriorityQueue).
     *
     * Return an int array with a capacity equal to the size of the list. The
     * returned array should have the elements in the list in sorted order.
     *
     * @param data the data to sort
     * @return the array with length equal to the size of the input list that
     * holds the elements from the list is sorted order
     * @throws java.lang.IllegalArgumentException if the data is null
     */
    public static int[] heapSort(List<Integer> data) {
        if (data == null) {
            throw new IllegalArgumentException("The list provided is null");
        }
        Queue<Integer> heap = new PriorityQueue<>(data);
        int[] sortedList = new int[data.size()];

        for (int a = 0; a < data.size(); a++) {
            sortedList[a] = heap.remove();
        }

        return sortedList;
    }
}
