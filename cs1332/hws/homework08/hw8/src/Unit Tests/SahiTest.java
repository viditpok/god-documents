import org.junit.Test;
import static org.junit.Assert.assertArrayEquals;


public class SahiTest {

    private static final int TIMEOUT = 200;

    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort1() {
        int[] unsortedArray = new int[]{Integer.MIN_VALUE, 54, 28, 58, 84, 20, 122, -85, 3};
        int[] sortedArray = new int[]{Integer.MIN_VALUE, -85, 3, 20, 28, 54, 58, 84, 122};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }
    
    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort2() {
        int[] unsortedArray = new int[]{Integer.MIN_VALUE, 54, Integer.MIN_VALUE, 58, 84, 20, 122, -85, Integer.MAX_VALUE, 3};
        int[] sortedArray = new int[]{Integer.MIN_VALUE, Integer.MIN_VALUE, -85, 3, 20, 54, 58, 84, 122, Integer.MAX_VALUE};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }
    
    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort3() {
        int[] unsortedArray = new int[]{Integer.MIN_VALUE, 54, 58, 84, 20, 122, -85, Integer.MAX_VALUE};
        int[] sortedArray = new int[]{Integer.MIN_VALUE, -85, 20, 54, 58, 84, 122, Integer.MAX_VALUE};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }
    
    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort4() {
        int[] unsortedArray = new int[]{Integer.MIN_VALUE};
        int[] sortedArray = new int[]{Integer.MIN_VALUE};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }
    
    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort5() {
        int[] unsortedArray = new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE};
        int[] sortedArray = new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }
    
    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort6() {
        int[] unsortedArray = new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE};
        int[] sortedArray = new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }
    
    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort7() {
        int[] unsortedArray = new int[]{Integer.MAX_VALUE, -90, Integer.MIN_VALUE, 54, 58, 84, 20, 122, -85, Integer.MAX_VALUE};
        int[] sortedArray = new int[]{Integer.MIN_VALUE, -90, -85, 20, 54, 58, 84, 122, Integer.MAX_VALUE, Integer.MAX_VALUE};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }
    
    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort8() {
        int[] unsortedArray = new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE,  Integer.MIN_VALUE, };
        int[] sortedArray = new int[]{Integer.MIN_VALUE, Integer.MIN_VALUE, Integer.MAX_VALUE};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }
    
    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort9() {
        int[] unsortedArray = new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE,  Integer.MAX_VALUE, };
        int[] sortedArray = new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }
    
    @Test(timeout = TIMEOUT)
    public void testLsdRadixSort10() {
        int[] unsortedArray = new int[]{Integer.MAX_VALUE, 1, 2, -1, 3, 4, 26, 0};
        int[] sortedArray = new int[]{-1, 0, 1, 2, 3, 4, 26, Integer.MAX_VALUE};
        Sorting.lsdRadixSort(unsortedArray);
        assertArrayEquals(sortedArray, unsortedArray);
    }
    
}
