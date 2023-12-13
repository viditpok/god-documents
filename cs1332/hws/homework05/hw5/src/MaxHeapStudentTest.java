import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.NoSuchElementException;

import static org.junit.Assert.*;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import java.util.ArrayList;
import java.util.NoSuchElementException;

import static org.junit.Assert.*;

/**
 * This is a basic set of unit tests for MaxHeap.
 *
 * Passing these tests doesn't guarantee any grade on these assignments. These
 * student JUnits that we provide should be thought of as a sanity check to
 * help you get started on the homework and writing JUnits in general.
 *
 * We highly encourage you to write your own set of JUnits for each homework
 * to cover edge cases you can think of for each data structure. Your code must
 * work correctly and efficiently in all cases, which is why it's important
 * to write comprehensive tests to cover as many cases as possible.
 *
 * @author CS 1332 TAs
 * @version 1.0
 */
public class MaxHeapStudentTest {
    private static final int TIMEOUT = 200;
    private MaxHeap<String> heap;

    @Before
    public void setUp() {
        heap = new MaxHeap<>();
    }

    @Test(timeout = TIMEOUT)
    public void testInitializationWithStrings() {
        ArrayList<String> data = new ArrayList<>();
        data.add("e");
        data.add("c");
        data.add("a");
        data.add("d");
        data.add("b");
        data.add("f");
        this.heap = new MaxHeap<>(data);

        /*
         *              f
         *             /  \
         *            d    e
         *           / \  /
         *          c  b  a
         *
         *         [null, 10, 5, 6, 2, 1, -1, null, null, null, null, null, null]
         *         Should be like above even though the data values are strings
         */
        String[] expected = new String[13];
        expected[1] = "f";
        expected[2] = "d";
        expected[3] = "e";
        expected[4] = "c";
        expected[5] = "b";
        expected[6]= "a";

        assertEquals(6, this.heap.size());
        assertArrayEquals(expected, this.heap.getBackingArray());
    }

    @Test(timeout = TIMEOUT)
    public void testAdd() {
        /*
                a
         */
        this.heap.add("a");
        assertEquals(1, this.heap.size());
        String[] expected = new String[13];
        expected[1] = "a";
        assertArrayEquals(expected, this.heap.getBackingArray());

        /*
                 b
                /
               a
         */
        this.heap.add("b");
        assertEquals(2, this.heap.size());
        expected[1] = "b";
        expected[2] = "a";
        assertArrayEquals(expected, this.heap.getBackingArray());

        /*
                c
               /  \
              a    b
         */
        this.heap.add("c");
        assertEquals(3, this.heap.size());
        expected[1] = "c";
        expected[2] = "a";
        expected[3] = "b";
        assertArrayEquals(expected, this.heap.getBackingArray());

        /*
                c
               /  \
              a    b
             /
            `
         */
        this.heap.add("`");
        assertEquals(4, this.heap.size());
        expected[1] = "c";
        expected[2] = "a";
        expected[3] = "b";
        expected[4] = "`";
        assertArrayEquals(expected, this.heap.getBackingArray());
    }

    @Test()
    public void testAddResize() {
        for (int i = 97; i < 109; i++) {
            this.heap.add(Character.toString((char) i));
        }
        assertEquals(12, this.heap.size());

        String[] expected = new String[13];
        expected[1] = "l";
        expected[2] = "j";
        expected[3] = "k";
        expected[4] = "g";
        expected[5] = "i";
        expected[6] = "f";
        expected[7] = "e";
        expected[8] = "a";
        expected[9] = "d";
        expected[10] = "c";
        expected[11] = "h";
        expected[12] = "b";

        assertArrayEquals(expected, this.heap.getBackingArray());

        this.heap.add("`");
        assertEquals(13, this.heap.size());

        expected = new String[26];
        expected[1] = "l";
        expected[2] = "j";
        expected[3] = "k";
        expected[4] = "g";
        expected[5] = "i";
        expected[6] = "f";
        expected[7] = "e";
        expected[8] = "a";
        expected[9] = "d";
        expected[10] = "c";
        expected[11] = "h";
        expected[12] = "b";
        expected[13] = "`";

        assertArrayEquals(expected, this.heap.getBackingArray());

        this.heap.clear();
        for (int i = 97; i < 109; i++) {
            this.heap.add(Character.toString((char) i));
        }
        assertEquals(12, this.heap.size());

        expected = new String[13];
        expected[1] = "l";
        expected[2] = "j";
        expected[3] = "k";
        expected[4] = "g";
        expected[5] = "i";
        expected[6] = "f";
        expected[7] = "e";
        expected[8] = "a";
        expected[9] = "d";
        expected[10] = "c";
        expected[11] = "h";
        expected[12] = "b";

        assertArrayEquals(expected, this.heap.getBackingArray());

        this.heap.add("m");
        assertEquals(13, this.heap.size());
        expected = new String[26];
        expected[1] = "m";
        expected[2] = "j";
        expected[3] = "l";
        expected[4] = "g";
        expected[5] = "i";
        expected[6] = "k";
        expected[7] = "e";
        expected[8] = "a";
        expected[9] = "d";
        expected[10] = "c";
        expected[11] = "h";
        expected[12] = "b";
        expected[13] = "f";
        assertArrayEquals(expected, this.heap.getBackingArray());

        this.heap.remove();
        assertEquals(12, this.heap.size());
        expected[1] = "l";
        expected[2] = "j";
        expected[3] = "k";
        expected[4] = "g";
        expected[5] = "i";
        expected[6] = "f";
        expected[7] = "e";
        expected[8] = "a";
        expected[9] = "d";
        expected[10] = "c";
        expected[11] = "h";
        expected[12] = "b";
        expected[13] = null;
        assertArrayEquals(expected, this.heap.getBackingArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveOneElement() {
        String temp = "1";
        this.heap.add(temp);
        assertEquals(1, this.heap.size());

        String[] expected = new String[13];
        assertEquals(temp, this.heap.remove());
        assertEquals(0, this.heap.size());
        assertArrayEquals(expected, this.heap.getBackingArray());
    }

    @Test(timeout = TIMEOUT)
    public void testGetMaxOneElement() {
        String temp = "a";
        this.heap.add(temp);

        assertEquals(1, this.heap.size());
        assertEquals(temp, this.heap.getMax());

        temp = "b";
        this.heap.add(temp);

        assertEquals(2, this.heap.size());
        assertEquals(temp, this.heap.getMax());
    }
    @Test(timeout = TIMEOUT)
    public void testInitializationExceptions() {
        ArrayList<String> data = null;
        Exception exception = Assert.assertThrows(IllegalArgumentException.class, () -> this.heap = new MaxHeap<>(null));
        assertNotNull(exception.getMessage());

        data = new ArrayList<>();
        data.add("1");
        data.add("2");
        data.add(null);
        data.add("4");
        ArrayList<String> finalData = data;
        exception = Assert.assertThrows(IllegalArgumentException.class, () -> this.heap = new MaxHeap<>(finalData));
        assertNotNull(exception.getMessage());
    }

    @Test(timeout = TIMEOUT)
    public void testAddException() {
        Exception exception = Assert.assertThrows(IllegalArgumentException.class, () -> this.heap.add(null));
        assertNotNull(exception.getMessage());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveException() {
        Exception exception = Assert.assertThrows(NoSuchElementException.class, () -> this.heap.remove());
        assertNotNull(exception.getMessage());
    }

    @Test(timeout = TIMEOUT)
    public void testGetMaxException() {
        Exception exception = Assert.assertThrows(NoSuchElementException.class, () -> this.heap.getMax());
        assertNotNull(exception.getMessage());
    }
}