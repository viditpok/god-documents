import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class MyTest {

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

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutChainedCorrect() {
        map.put(1, "A");
        map.put(6, "B");
        map.put(2, "F");
        
        /*
        The HashMap should be:
        table[1]: <1, "A"> -> null
        table[2]: <2, "F"> -> null
        table[6]: <6, "B"> -> null
        Other indices are null.
         */
        assertEquals(map.getTable()[2], new ExternalChainingMapEntry<>(2, "F"));
        assertEquals(map.getTable()[1].getNext(), new ExternalChainingMapEntry<>(1, "A"));
        assertEquals(map.getTable()[1], new ExternalChainingMapEntry<>(6, "B"));
        assertEquals(3, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutResizeCorrect() {
        map.put(1, "A");
        map.put(6, "B");
        map.put(2, "F");
        map.put(4, "Q");
        map.put(1, "G"); // This shouldn't change size
        map.put(7, "Q");
        /*
        The HashMap should be:
        table[1]: <1, "G"> -> null
        table[2]: <2, "F"> -> null
        table[4]: <4, "Q"> -> null
        table[6]: <6, "B"> -> null
        table[7]: <7, "Q"> -> null
        Other indices are null.
         */
        assertEquals(11, map.getTable().length);
        assertEquals(new ExternalChainingMapEntry<>(1, "G"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(6, "B"), map.getTable()[6]);
        assertEquals(new ExternalChainingMapEntry<>(2, "F"), map.getTable()[2]);
        assertEquals(new ExternalChainingMapEntry<>(4, "Q"), map.getTable()[4]);
        assertEquals(new ExternalChainingMapEntry<>(7, "Q"), map.getTable()[7]);
        assertEquals(5, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testReplaceOldValue1() {
        // Replace while resizing
        map.put(1, "A");
        map.put(6, "B");
        map.put(2, "F");
        map.put(4, "Q");
        map.put(1, "G"); // This shouldn't change size
        map.put(2, "Q");
        /*
        The HashMap should be:
        table[1]: <1, "G"> -> null
        table[2]: <2, "Q"> -> null
        table[4]: <4, "Q"> -> null
        table[6]: <6, "B"> -> null
        Other indices are null.
         */
        assertEquals(11, map.getTable().length);
        assertEquals(new ExternalChainingMapEntry<>(1, "G"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(6, "B"), map.getTable()[6]);
        assertEquals(new ExternalChainingMapEntry<>(2, "Q"), map.getTable()[2]);
        assertEquals(new ExternalChainingMapEntry<>(4, "Q"), map.getTable()[4]);
        assertEquals(4, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testReplaceOldValue2() {
        // Simple replacement
        map.put(1, "A");
        map.put(1, "G");
        map.put(1, "K");
        map.put(1, "J");
                /*
        The HashMap should be:
        table[1]: <1, "J"> -> null
        Other indices are null.
         */
        assertEquals(5, map.getTable().length);
        assertEquals(new ExternalChainingMapEntry<>(1, "J"), map.getTable()[1]);
        assertEquals(1, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testReplaceOldValue3() {
        // Need to traverse linkedlist to find duplicates
        map.put(1, "A");
        map.put(56, "G");
        map.put(166, "K");
        map.put(221, "J");
        map.put(56, "BF");
        map.put(1, "LOL");
        map.put(221, "Cool");
        /*
        The HashMap should be:
        table[1]: <221, "Cool"> -> <1, "LOL"> -> <56, "BF"> -> <166, "K"> -> null
        Other indices are null.
         */
        assertEquals(11, map.getTable().length);
        assertEquals(new ExternalChainingMapEntry<>(221, "Cool"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(1, "LOL"),
                map.getTable()[1].getNext());
        assertEquals(new ExternalChainingMapEntry<>(56, "BF"),
                map.getTable()[1].getNext().getNext());
        assertEquals(new ExternalChainingMapEntry<>(166, "K"),
                map.getTable()[1].getNext().getNext().getNext());
        assertEquals(4, map.size());
    }

   @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void putNewEntry1() {
        // Direct insertion
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        /*
        The HashMap should be:
        table[1]: <1, "A"> -> null
        table[2]: <2, "B"> -> null
        table[3]: <3, "C"> -> null
        table[3]: <4, "D"> -> null
        Other indices are null.
         */

        assertEquals(11, map.getTable().length);
        assertEquals(new ExternalChainingMapEntry<>(1, "A"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(2, "B"),
                map.getTable()[2]);
        assertEquals(new ExternalChainingMapEntry<>(3, "C"),
                map.getTable()[3]);
        assertEquals(new ExternalChainingMapEntry<>(4, "D"),
                map.getTable()[4]);
        assertEquals(4, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void putNewEntry2() {

        // Direct insertion
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");

        assertEquals(11, map.getTable().length);
        assertEquals(4, map.size());

        // Need to traverse through linked list.
        map.put(12, "A");
        assertEquals(new ExternalChainingMapEntry<>(12, "A"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(1, "A"), map.getTable()[1].getNext());
        assertEquals(5, map.size());

        map.put(56, "A");
        assertEquals(new ExternalChainingMapEntry<>(56, "A"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(12, "A"), map.getTable()[1].getNext());
        assertEquals(new ExternalChainingMapEntry<>(1, "A"), map.getTable()[1].getNext().getNext());
        assertEquals(6, map.size());

        /*
        The final HashMap should be:
        table[1]: <56, "A"> -> <12, "A"> -> <1, "A"> -> null
        table[2]: <2, "B"> -> null
        table[3]: <3, "C"> -> null
        table[3]: <4, "D"> -> null
        Other indices are null.
         */
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void putNewEntry3() {
        // Direct insertion
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(Integer.MAX_VALUE, "MAX");
        map.put(Integer.MIN_VALUE, "MIN");
        /*
        The HashMap should be:
        table[1]: <Integer.MAX_VALUE, "MAX"> -> <1, "A"> -> null
        table[2]: <Integer.MIN_VALUE, "MIN"> -> <2, "B"> -> null
        table[3]: <3, "C"> -> null
        table[3]: <4, "D"> -> null
        Other indices are null.
         */

        assertEquals(11, map.getTable().length);
        assertEquals(new ExternalChainingMapEntry<>(Integer.MAX_VALUE, "MAX"),
                map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(1, "A"), map.getTable()[1].getNext());
        assertEquals(new ExternalChainingMapEntry<>(Integer.MIN_VALUE, "MIN"),
                map.getTable()[2]);
        assertEquals(new ExternalChainingMapEntry<>(2, "B"),
                map.getTable()[2].getNext());
        assertEquals(new ExternalChainingMapEntry<>(3, "C"),
                map.getTable()[3]);
        assertEquals(new ExternalChainingMapEntry<>(4, "D"),
                map.getTable()[4]);
        assertEquals(6, map.size());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testPutException() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(null, "Good");
        });
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(1, null);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(null, null);
        });
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void removeNotFound1() {

        // Simple not found
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        /*
        The HashMap should be:
        table[1]: <1, "A"> -> null
        table[2]: <2, "B"> -> null
        table[3]: <3, "C"> -> null
        table[3]: <4, "D"> -> null
        Other indices are null.
         */
        assertEquals(11, map.getTable().length);
        assertEquals(4, map.size());

        assertThrows(NoSuchElementException.class, () -> {
            map.remove(5);
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.remove(0);
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.remove(9);
        });
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void removeNotFound2() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(15, "C");
        map.put(12, "A");
        map.put(34, "A");
        /*
        The final HashMap should be:
        table[1]: <34, "A"> -> <12, "A"> -> <1, "A"> -> null
        table[2]: <2, "B"> -> null
        table[3]: <3, "C"> -> null
        table[3]: <15, "C"> -> <4, "D"> -> null
        Other indices are null.
         */
        assertEquals(11, map.getTable().length);
        assertEquals(7, map.size());

        // Need to traverse linkedlist
        assertThrows(NoSuchElementException.class, () -> {
            map.remove(23);
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.remove(45);
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.remove(14);
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.remove(26);
        });

        assertEquals(7, map.size());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void remove1() {
        // Simple remove
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        /*
        The HashMap before removal should be:
        table[1]: <1, "A"> -> null
        table[2]: <2, "B"> -> null
        table[3]: <3, "C"> -> null
        table[3]: <4, "D"> -> null
        Other indices are null.
         */
        assertEquals("A", map.remove(1));
        assertEquals("D", map.remove(4));

        assertEquals(2, map.size());
        assertNull(map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(2, "B"), map.getTable()[2]);
        assertEquals(new ExternalChainingMapEntry<>(3, "C"), map.getTable()[3]);
        assertNull(map.getTable()[4]);
        /*
        The HashMap after removal should be:
        table[2]: <2, "B"> -> null
        table[3]: <3, "C"> -> null
        Other indices are null.
         */
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void remove2() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(15, "C");
        map.put(12, "A'");
        map.put(34, "A''");
        map.put(45, "A'''");
        map.put(3, "C");
        /*
        The final HashMap should be:
        table[1]: <45, "A'''"> -> <34, "A''"> -> <12, "A'"> -> <1, "A"> -> null
        table[2]: <2, "B"> -> null
        table[3]: <3, "C"> -> null
        table[3]: <15, "C"> -> null
        Other indices are null.
         */

        assertEquals(11, map.getTable().length);
        assertEquals(7, map.size());

        // Reorder elements after resize
        // Sorry for confusion in size since this will cause size = actualSize - 1
        map.remove(1);
        map.getTable()[1].getNext().getNext().setNext(new ExternalChainingMapEntry<>(1, "A"));

        // remove in the middle
        /*
        The HashMap after this removal should be:
        table[1]: <45, "A'''"> -> <12, "A'"> -> <1, "A"> -> null
        table[2]: <2, "B"> -> null
        table[3]: <3, "C"> -> null
        table[3]: <15, "C"> -> null
        Other indices are null.
         */
        assertEquals("A''", map.remove(34));
        assertEquals(new ExternalChainingMapEntry<>(45, "A'''"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(12, "A'"), map.getTable()[1].getNext());
        assertEquals(new ExternalChainingMapEntry<>(1, "A"), map.getTable()[1].getNext().getNext());
        assertNull(map.getTable()[1].getNext().getNext().getNext());
        assertEquals(5, map.size());

        // remove the last element
        /*
        The HashMap after this removal should be:
        table[1]: <45, "A'''"> -> <12, "A'">  -> null
        table[2]: <2, "B"> -> null
        table[3]: <3, "C"> -> null
        table[3]: <15, "C"> -> null
        Other indices are null.
         */
        assertEquals("A", map.remove(1));
        assertEquals(new ExternalChainingMapEntry<>(45, "A'''"), map.getTable()[1]);
        assertEquals(new ExternalChainingMapEntry<>(12, "A'"), map.getTable()[1].getNext());
        assertNull(map.getTable()[1].getNext().getNext());
        assertEquals(4, map.size());

        // remove the front
        /*
        The HashMap after this removal should be:
        table[1]: <12, "A'">  -> null
        table[2]: <2, "B"> -> null
        table[3]: <3, "C"> -> null
        table[3]: <15, "C"> -> null
        Other indices are null.
         */
        assertEquals("A'''", map.remove(45));
        assertEquals(new ExternalChainingMapEntry<>(12, "A'"), map.getTable()[1]);
        assertNull(map.getTable()[1].getNext());
        assertEquals(3, map.size());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testRemoveException() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.remove(null);
        });
    }

    @Test
    @SuppressWarnings("unchecked")
    public void get1() {
        // Simple get
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");

        assertEquals("A", map.get(1));
        assertEquals("D", map.get(4));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void get2() {
        // Traverse linkedlist to get
        map.put(1, "A");
        map.put(2, "B");
        map.put(15, "C");
        map.put(12, "A'");
        map.put(34, "A''");
        map.put(45, "A'''");
        map.put(3, "C");
        /*
        The HashMap should be:
        table[1]: <45, "A'''"> -> <12, "A'"> -> <1, "A"> -> null
        table[2]: <2, "B"> -> null
        table[3]: <3, "C"> -> null
        table[3]: <15, "C"> -> null
        Other indices are null.
         */
        assertEquals("A'", map.get(12));
        assertEquals("A'''", map.get(45));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void get3() {
        // Exceptions
        map.put(1, "A");
        map.put(2, "B");
        map.put(15, "C");
        map.put(12, "A'");
        map.put(34, "A''");
        map.put(45, "A'''");
        map.put(3, "C");

        assertThrows(NoSuchElementException.class, () -> {
            map.get(23);
        });
        assertThrows(NoSuchElementException.class, () -> {
            map.get(4);
        });
        assertThrows(NoSuchElementException.class, () -> {
            map.get(56);
        });
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetException() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.get(null);
        });
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testContains() {
        // Exceptions
        map.put(1, "A");
        map.put(2, "B");
        map.put(15, "C");
        map.put(12, "A'");
        map.put(34, "A''");
        map.put(45, "A'''");
        map.put(3, "C");

        assertTrue(map.containsKey(1));
        assertTrue(map.containsKey(15));
        assertTrue(map.containsKey(12));
        assertTrue(map.containsKey(34));
        assertTrue(map.containsKey(45));
        assertFalse(map.containsKey(10));
        assertFalse(map.containsKey(0));
        assertFalse(map.containsKey(56));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testContainsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.containsKey(null);
        });
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testKeySet() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(15, "C");
        map.put(12, "A'");
        map.put(34, "A''");
        map.put(45, "A'''");
        map.put(3, "C");
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
        map.put(1, "A");
        map.put(2, "B");
        map.put(15, "C'");
        map.put(12, "A'");
        map.put(34, "A''");
        map.put(45, "A'''");
        map.put(3, "C");
        List<String> list = Arrays.asList("A'''", "A''", "A'", "A", "B", "C", "C'");

        // order must be the same
        assertEquals(list, map.values());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testClear() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(15, "C'");
        map.put(12, "A'");
        map.put(34, "A''");
        map.put(45, "A'''");
        map.put(3, "C");
        map.clear();

        assertArrayEquals(new ExternalChainingMapEntry[13], map.getTable());
        assertEquals(0, map.size());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void resizeException() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(15, "C'");
        map.put(12, "A'");
        map.put(34, "A''");
        map.put(45, "A'''");
        map.put(3, "C");
        assertThrows(IllegalArgumentException.class, () -> {
            map.resizeBackingTable(-1);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            map.resizeBackingTable(map.size() - 1);
        });
    }
}