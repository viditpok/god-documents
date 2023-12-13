import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class MyTestCases {
    private static final int TIMEOUT = 200;
    private ExternalChainingHashMap<Integer, String> map;

    @Before
    public void setUp() {
        map = new ExternalChainingHashMap<>(5);
    }

    @Test(timeout = TIMEOUT)
    public void testInitialization() {
        assertEquals(0, map.size());
        assertArrayEquals(new ExternalChainingMapEntry[5], map.getTable());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testPutException() {
        //Tests the three possible exceptions for the put method.
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(null, null);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(null, "Testing");
        });
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(1, null);
        });
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutWithoutResize() {
        map.put(1, "First");
        map.put(6, "Second");
        map.put(11, "Third");

        /*
        The HashMap:
        table[1]: <11, "Third"> -> <6, "Second"> -> <1, "First"> -> null
        All other indices are null.
        No resize occurred.
         */
        assertEquals(map.getTable()[1], new ExternalChainingMapEntry<>(11, "Third"));
        assertEquals(map.getTable()[1].getNext().getNext(), new ExternalChainingMapEntry<>(1, "First"));
        assertEquals(map.getTable()[1].getNext(), new ExternalChainingMapEntry<>(6, "Second"));
        assertEquals(3, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testSimpleNewPut() {
        // Simply adds 4 elements and does 1 resize
        map.put(1, "one");
        map.put(2, "two");
        map.put(3, "three");
        map.put(4, "four");
        /*
        The HashMap:
        table[1]: <1, "one"> -> null
        table[2]: <2, "two"> -> null
        table[3]: <3, "three"> -> null
        table[3]: <4, "four"> -> null
        All other indices are null.
         */
        assertEquals(new ExternalChainingMapEntry<>(1, "one"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(2, "two"),
                map.getTable()[2]);
        assertEquals(new ExternalChainingMapEntry<>(3, "three"),
                map.getTable()[3]);
        assertEquals(new ExternalChainingMapEntry<>(4, "four"),
                map.getTable()[4]);
        assertEquals(11, map.getTable().length);
        assertEquals(4, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutWithResize() {
        //Tests the resize and a basic replacement separately.
        map.put(1, "First");
        map.put(6, "Second");
        map.put(2, "Third");
        map.put(4, "Fourth");
        map.put(1, "Fifth"); // replaces and return "First"
        map.put(7, "Sixth"); // Causes the resize
        /*
        The HashMap:
        table[1]: <1, "Fifth"> -> null
        table[2]: <2, "Third"> -> null
        table[4]: <4, "Fourth"> -> null
        table[6]: <6, "Second"> -> null
        table[7]: <7, "Sixth"> -> null
        All other indices are null.
         */

        assertEquals(new ExternalChainingMapEntry<>(1, "Fifth"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(6, "Second"), map.getTable()[6]);
        assertEquals(new ExternalChainingMapEntry<>(2, "Third"), map.getTable()[2]);
        assertEquals(new ExternalChainingMapEntry<>(4, "Fourth"), map.getTable()[4]);
        assertEquals(new ExternalChainingMapEntry<>(7, "Sixth"), map.getTable()[7]);
        assertEquals(11, map.getTable().length);
        assertEquals(5, map.size());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testResizeException() {
        map.put(1, "one");
        map.put(2, "two");
        map.put(4, "four");
        map.put(8, "eight");
        assertThrows(IllegalArgumentException.class, () -> {
            map.resizeBackingTable(map.size() - 1);
        }); //Tests resizing backing array smaller than the size.
        assertThrows(IllegalArgumentException.class, () -> {
            map.resizeBackingTable(-1);
        }); //Tests resizing backing array to a negative number.
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testDoubleResize() {
        // tests if resize works for double resize
        map.put(1, "data1");
        map.put(2, "data2");
        map.put(3, "data3");
        map.put(4, "data4");
        map.put(5, "data5");
        map.put(6, "data6");
        map.put(7, "data7");
        map.put(8, "data8");
        map.put(9, "data9");
        map.put(10, "data10");
        /*
        The size of the hashmap should be 10
        The length of the hashmap should be 23
            5 -> 11 -> 23

        table[1] through table[10] should have data in them
        Every other index should be null.
         */
        assertEquals(new ExternalChainingMapEntry<>(1, "data1"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(2, "data2"), map.getTable()[2]);
        assertEquals(new ExternalChainingMapEntry<>(3, "data3"), map.getTable()[3]);
        assertEquals(new ExternalChainingMapEntry<>(4, "data4"), map.getTable()[4]);
        assertEquals(new ExternalChainingMapEntry<>(5, "data5"), map.getTable()[5]);
        assertEquals(new ExternalChainingMapEntry<>(6, "data6"), map.getTable()[6]);
        assertEquals(new ExternalChainingMapEntry<>(7, "data7"), map.getTable()[7]);
        assertEquals(new ExternalChainingMapEntry<>(8, "data8"), map.getTable()[8]);
        assertEquals(new ExternalChainingMapEntry<>(9, "data9"), map.getTable()[9]);
        assertEquals(new ExternalChainingMapEntry<>(10, "data10"), map.getTable()[10]);
        assertEquals(23, map.getTable().length);
        assertEquals(10, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testTraverseLinkedListPut() {

        // Simple put methods that will resize
        map.put(1, "one");
        map.put(2, "two");
        map.put(3, "three");
        map.put(4, "four");
        assertEquals(11, map.getTable().length);
        assertEquals(4, map.size());

        // now traverses linked list
        map.put(12, "twelve");
        assertEquals(new ExternalChainingMapEntry<>(12, "twelve"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(1, "one"), map.getTable()[1].getNext());
        assertEquals(5, map.size());

        map.put(56, "fifty six");
        assertEquals(new ExternalChainingMapEntry<>(56, "fifty six"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(12, "twelve"), map.getTable()[1].getNext());
        assertEquals(new ExternalChainingMapEntry<>(1, "one"), map.getTable()[1].getNext().getNext());
        assertEquals(6, map.size());

        /*
        The HashMap:
        table[1]: <56, "fifty six"> -> <12, "twelve"> -> <1, "one"> -> null
        table[2]: <2, "two"> -> null
        table[3]: <3, "three"> -> null
        table[3]: <4, "four"> -> null
        All other indices are null.
         */
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testTraverseLinkedListReplace() {
        // Needs to traverse a linkedlist to find duplicates
        map.put(1, "First");
        map.put(56, "Second");
        map.put(166, "Third");
        map.put(221, "Fourth");
        map.put(56, "Second replaced");
        map.put(1, "First replaced");
        map.put(221, "Fourth replaced");
        /*
        The HashMap:
        table[1]: <221, "Fourth replaced"> -> <166, "Third"> -> <56, "Second replaced"> -> <1, "First replaced"> -> null
        All other indices are null.
         */

        assertEquals(new ExternalChainingMapEntry<>(221, "Fourth replaced"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(1, "First replaced"),
                map.getTable()[1].getNext());
        assertEquals(new ExternalChainingMapEntry<>(56, "Second replaced"),
                map.getTable()[1].getNext().getNext());
        assertEquals(new ExternalChainingMapEntry<>(166, "Third"),
                map.getTable()[1].getNext().getNext().getNext());
        assertEquals(11, map.getTable().length);
        assertEquals(4, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutSimultaneousReplaceAndResize() {
        map.put(1, "getting replaced");
        map.put(6, "six");
        map.put(2, "getting replaced");
        map.put(4, "four");
        map.put(1, "one"); // Replaces getting replaced with key 1
        map.put(2, "two"); // Replaced getting replaced with key 2
        /*
        The HashMap:
        table[1]: <1, "one"> -> null
        table[2]: <2, "two"> -> null
        table[4]: <4, "four"> -> null
        table[6]: <6, "six"> -> null
        All other indices are null.
         */
        assertEquals(new ExternalChainingMapEntry<>(1, "one"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(6, "six"), map.getTable()[6]);
        assertEquals(new ExternalChainingMapEntry<>(2, "two"), map.getTable()[2]);
        assertEquals(new ExternalChainingMapEntry<>(4, "four"), map.getTable()[4]);
        assertEquals(11, map.getTable().length);
        assertEquals(4, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutWithMinAndMax() {
        // Direct insertion
        map.put(1, "one");
        map.put(2, "two");
        map.put(3, "three");
        map.put(4, "four");
        map.put(Integer.MAX_VALUE, "MAX");
        map.put(Integer.MIN_VALUE, "MIN");
        /*
        The HashMap:
        table[1]: <Integer.MAX_VALUE, "MAX"> -> <1, "one"> -> null
        table[2]: <Integer.MIN_VALUE, "MIN"> -> <2, "two"> -> null
        table[3]: <3, "three"> -> null
        table[3]: <4, "four"> -> null
        All other indices are null.
         */


        assertEquals(new ExternalChainingMapEntry<>(Integer.MAX_VALUE, "MAX"),
                map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(1, "one"), map.getTable()[1].getNext());
        assertEquals(new ExternalChainingMapEntry<>(Integer.MIN_VALUE, "MIN"),
                map.getTable()[2]);
        assertEquals(new ExternalChainingMapEntry<>(2, "two"),
                map.getTable()[2].getNext());
        assertEquals(new ExternalChainingMapEntry<>(3, "three"),
                map.getTable()[3]);
        assertEquals(new ExternalChainingMapEntry<>(4, "four"),
                map.getTable()[4]);
        assertEquals(11, map.getTable().length);
        assertEquals(6, map.size());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testRemoveNullException() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.remove(null);
        });
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testRemoveNoSuchElementException() {
        map.put(1, "one");
        map.put(2, "two");
        map.put(3, "three");
        assertEquals(3, map.size());
        assertEquals(5, map.getTable().length);
        assertThrows(NoSuchElementException.class, () -> {
            map.remove(4);
        });
        assertThrows(NoSuchElementException.class, () -> {
            map.remove(0);
        });
        assertThrows(NoSuchElementException.class, () -> {
            map.remove(1234);
        });
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveNoSuchElementExceptionTraversal() {
        //searches through a linked list to find that the element does not exist
        map.put(1, "one");
        map.put(2, "two");
        map.put(4, "four");
        map.put(15, "fifteen");
        map.put(12, "twelve");
        map.put(34, "thirty four");
        /*
        The HashMap:
        table[1]: <34, "thirty four"> -> <12, "twelve"> -> <1, "one"> -> null
        table[2]: <2, "two"> -> null
        table[3]: <15, "fifteen"> -> <4, "four"> -> null
        All other indices are null.
         */
        assertEquals(11, map.getTable().length);
        assertEquals(6, map.size());

        assertThrows(NoSuchElementException.class, () -> {
            map.remove(14);
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.remove(45);
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.remove(23);
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.remove(26);
        });

        assertEquals(6, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testSimpleRemove() {
        // Simple remove
        map.put(1, "one");
        map.put(2, "two");
        map.put(3, "three");
        map.put(4, "four");
        /*
        Hashmap before remove:
        table[1]: <1, "one"> -> null
        table[2]: <2, "two"> -> null
        table[3]: <3, "three"> -> null
        table[3]: <4, "four"> -> null
        All other indices are null.
         */

        assertEquals(4, map.size());
        assertEquals(11, map.getTable().length);

        assertEquals("one", map.remove(1));
        assertEquals("four", map.remove(4));

        assertEquals(2, map.size());
        assertNull(map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(2, "two"), map.getTable()[2]);
        assertEquals(new ExternalChainingMapEntry<>(3, "three"), map.getTable()[3]);
        assertNull(map.getTable()[4]);
        /*
        Final Hashmap:
        table[2]: <2, "two"> -> null
        table[3]: <3, "three"> -> null
        All other indices are null.
         */
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testTraversalRemovals() {
        //Tests for the remove method on the first, middle, and last elements in a linked list
        map.put(1, "one");
        map.put(2, "two");
        map.put(15, "fifteen");
        map.put(12, "twelve");
        map.put(34, "thirty four");
        map.put(45, "forty five");
        map.put(3, "three");
        /*
        The HashMap:
        table[1]: <45, "forty five"> -> <34, "thirty four"> -> <12, "twelve"> -> <1, "one"> -> null
        table[2]: <2, "two"> -> null
        table[3]: <3, "three"> -> null
        table[4]: <15, "fifteen"> -> null
        All other indices are null.
         */

        assertEquals(11, map.getTable().length);
        assertEquals(7, map.size());
        //map.remove(1);
        //map.getTable()[1].getNext().getNext().setNext(new ExternalChainingMapEntry<>(1, "one"));

        assertEquals("thirty four", map.remove(34));

        /*
        The HashMap after removing key 34:
        table[1]: <45, "forty five"> -> <12, "twelve"> -> <1, "one"> -> null
        table[2]: <2, "two"> -> null
        table[3]: <3, "three"> -> null
        table[4]: <15, "fifteen"> -> null
        All other indices are null.
        */

        assertEquals(new ExternalChainingMapEntry<>(45, "forty five"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(12, "twelve"), map.getTable()[1].getNext());
        assertEquals(new ExternalChainingMapEntry<>(1, "one"), map.getTable()[1].getNext().getNext());
        assertNull(map.getTable()[1].getNext().getNext().getNext());
        assertEquals(6, map.size());
        assertEquals("one", map.remove(1));

        /*
        The HashMap after removing key 1:
        table[1]: <45, "forty five"> -> <12, "twelve"> -> null
        table[2]: <2, "two"> -> null
        table[3]: <3, "three"> -> null
        table[4]: <15, "fifteen"> -> null
        All other indices are null.
        */

        assertEquals(new ExternalChainingMapEntry<>(45, "forty five"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(12, "twelve"), map.getTable()[1].getNext());
        assertNull(map.getTable()[1].getNext().getNext());
        assertEquals(5, map.size());

        /*
        The HashMap after removing key 45:
        table[1]:  <12, "twelve"> -> null
        table[2]: <2, "two"> -> null
        table[3]: <3, "three"> -> null
        table[4]: <15, "fifteen"> -> null
        All other indices are null.
        */
        assertEquals("forty five", map.remove(45));
        assertEquals(new ExternalChainingMapEntry<>(12, "twelve"), map.getTable()[1]);
        assertNull(map.getTable()[1].getNext());
        assertEquals(4, map.size());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.get(null);
        });
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetNoSuchElementException() {
        map.put(1, "one");
        assertThrows(NoSuchElementException.class, () -> {
            map.get(4);
        });
        assertThrows(NoSuchElementException.class, () -> {
            map.get(0);
        });
        assertThrows(NoSuchElementException.class, () -> {
            map.get(123);
        });
        assertThrows(NoSuchElementException.class, () -> {
            map.get(2);
        });
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetNoSuchElementException2() {
        map.put(1, "one");
        map.put(2, "two");
        map.put(15, "fifteen");
        map.put(12, "twelve");
        map.put(34, "thirty four");
        map.put(45, "forty five");
        map.put(3, "three");

        assertThrows(NoSuchElementException.class, () -> {
            map.get(0);
        });
        assertThrows(NoSuchElementException.class, () -> {
            map.get(4);
        });
        assertThrows(NoSuchElementException.class, () -> {
            map.get(5);
        });
        assertThrows(NoSuchElementException.class, () -> {
            map.get(1234);
        });
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testSimpleGet() {
        // Simple get
        map.put(1, "one");
        map.put(2, "two");
        map.put(3, "three");
        map.put(4, "four");

        assertEquals("two", map.get(2));
        assertEquals("three", map.get(3));
        assertEquals(11, map.getTable().length);
        assertEquals(4, map.size());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testTraversalGet() {
        map.put(1, "one");
        map.put(15, "fifteen");
        map.put(12, "twelve");
        map.put(34, "thirty four");
        map.put(45, "forty five");
        /*
        The HashMap:
        table[1]: <45, "forty five"> -> <12, "twelve"> -> <1, "one"> -> null
        table[4]: <15, "fifteen"> -> null
        All other indices are null.
         */
        assertEquals(5, map.size());
        assertEquals(11, map.getTable().length);
        assertEquals("twelve", map.get(12));
        assertEquals("forty five", map.get(45));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testContainsIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.containsKey(null);
        });
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testContains() {
        map.put(1, "one");
        map.put(2, "two");
        map.put(15, "fifteen");
        map.put(12, "twelve");
        map.put(34, "thirty four");
        map.put(45, "forty five");

        assertTrue(map.containsKey(1));
        assertTrue(map.containsKey(2));
        assertTrue(map.containsKey(12));
        assertTrue(map.containsKey(15));
        assertTrue(map.containsKey(45));
        assertFalse(map.containsKey(0));
        assertFalse(map.containsKey(3));
        assertFalse(map.containsKey(4));
        assertFalse(map.containsKey(5));
        assertFalse(map.containsKey(6));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testKeySet() {
        map.put(1, "one");
        map.put(2, "two");
        map.put(15, "fifteen");
        map.put(12, "twelve");
        map.put(34, "thirty four");
        map.put(45, "forty five");
        map.put(3, "three");
        Set<Integer> set = new HashSet<>(Arrays.asList(1, 34, 2, 3, 12, 45, 15));
        assertEquals(set, map.keySet());
        map.remove(15);
        set.remove(15);
        assertEquals(set, map.keySet());
        map.remove(1);
        set.remove(1);
        assertEquals(set, map.keySet());
        map.remove(45);
        set.remove(45);
        assertEquals(set, map.keySet());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testValues() {
        map.put(1, "one");
        map.put(2, "two");
        map.put(15, "fifteen");
        map.put(12, "twelve");
        map.put(34, "thirty four");
        map.put(45, "forty five");
        map.put(3, "three");
        List<String> list = Arrays.asList("forty five", "thirty four", "twelve", "one", "two", "three", "fifteen");
        assertEquals(list, map.values());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testClear() {
        map.put(1, "one");
        map.put(2, "two");
        map.put(15, "fifteen");
        map.put(12, "twelve");
        map.put(34, "thirty four");
        map.put(45, "forty five");
        map.put(3, "three");
        map.clear();
        assertEquals(13, map.getTable().length);
        assertEquals(0, map.size());
    }

}