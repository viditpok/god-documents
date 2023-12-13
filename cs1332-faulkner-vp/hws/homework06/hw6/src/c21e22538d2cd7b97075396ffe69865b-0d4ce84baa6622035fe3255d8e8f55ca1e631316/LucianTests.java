import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

/**
 * External Chaining HashMap JUnit Tests
 *
 * @author Lucian Tash
 * @version 1.0
 */
public class LucianTests {

    private static final int TIMEOUT = 200;
    private ExternalChainingHashMap<Integer, String> map;


    @Test(timeout = TIMEOUT)
    public void testDefaultConstructor() {
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        assertEquals(0, map.size());
        assertEquals(ExternalChainingHashMap.INITIAL_CAPACITY, map.getTable().length);
        assertArrayEquals(new ExternalChainingMapEntry[ExternalChainingHashMap.INITIAL_CAPACITY], map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testConstructor() {
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>(2112);
        assertEquals(0, map.size());
        assertEquals(2112, map.getTable().length);
        assertArrayEquals(new ExternalChainingMapEntry[2112], map.getTable());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPut() {

        // Collision = entries with different keys that hash to the same location
        // Replacement = entries with the same key; the new value replaces the old
        // Resize = backing array (table) regrows when load factor is exceeded

        // Small hashmap for basic tests (NO RESIZE)
        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[6];
        map = new ExternalChainingHashMap<>(6);     // [null, null, null, null, null, null]
        assertArrayEquals(expected, map.getTable());
        assertEquals(6, map.getTable().length);
        assertNull(map.put(0, "A"));                       // [<0 "A">, null, null, null, null, null]
        expected[0] = new ExternalChainingMapEntry(0, "A");
        assertArrayEquals(expected, map.getTable());
        assertEquals(6, map.getTable().length);
        assertNull(map.put(9, "D"));                       // [<0 "A">, null, null, <9 "D">, null, null]
        expected[3] = new ExternalChainingMapEntry(9, "D");
        assertArrayEquals(expected, map.getTable());
        assertEquals(6, map.getTable().length);
        assertNull(map.put(2, "B"));                       // [<0 "A">, null, <2 "B">, <9 "D">, null, null]
        expected[2] = new ExternalChainingMapEntry(2, "B");
        assertArrayEquals(expected, map.getTable());
        assertEquals(6, map.getTable().length);
        assertNull(map.put(5, "C"));                       // [<0 "A">, null, <2 "B">, <9 "D">, null, <5 "C">]
        expected[5] = new ExternalChainingMapEntry(5, "C");
        assertArrayEquals(expected, map.getTable());
        assertEquals(6, map.getTable().length);

        // Testing only resize (no collision or replacement)
        map = new ExternalChainingHashMap<>();
        assertEquals(13, map.getTable().length);
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));
        assertNull(map.put(7, "H"));
        assertNull(map.put(8, "I"));
        assertNull(map.put(9, "J"));
        assertNull(map.put(10, "K"));
        assertNull(map.put(11, "L"));
        assertNull(map.put(12, "M"));
        assertNull(map.put(13, "N"));
        assertNull(map.put(14, "O"));
        assertNull(map.put(15, "P"));
        assertNull(map.put(16, "Q"));
        assertEquals(27, map.getTable().length);
        assertEquals(17, map.size());
        expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[27];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");
        expected[5] = new ExternalChainingMapEntry<>(5, "F");
        expected[6] = new ExternalChainingMapEntry<>(6, "G");
        expected[7] = new ExternalChainingMapEntry<>(7, "H");
        expected[8] = new ExternalChainingMapEntry<>(8, "I");
        expected[9] = new ExternalChainingMapEntry<>(9, "J");
        expected[10] = new ExternalChainingMapEntry<>(10, "K");
        expected[11] = new ExternalChainingMapEntry<>(11, "L");
        expected[12] = new ExternalChainingMapEntry<>(12, "M");
        expected[13] = new ExternalChainingMapEntry<>(13, "N");
        expected[14] = new ExternalChainingMapEntry<>(14, "O");
        expected[15] = new ExternalChainingMapEntry<>(15, "P");
        expected[16] = new ExternalChainingMapEntry<>(16, "Q");
        assertArrayEquals(expected, map.getTable());

        // Testing only replacement (no collision or resize)
        map = new ExternalChainingHashMap<>();
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertEquals("A", map.put(0, "C"));
        assertEquals("C", map.put(0, "D"));
        assertNull(map.put(3, "E"));
        assertEquals(13, map.getTable().length);
        assertEquals(3, map.size());
        expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(0, "D");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "E");
        assertArrayEquals(expected, map.getTable());

        // Testing only collision (no replacement or resize)
        map = new ExternalChainingHashMap<>();
        assertNull(map.put(0, "A"));
        assertNull(map.put(14, "B"));
        assertNull(map.put(13, "C"));
        assertNull(map.put(26, "D"));
        assertNull(map.put(3, "E"));
        assertEquals(13, map.getTable().length);
        assertEquals(5, map.size());
        expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "D");
        expected[1] = new ExternalChainingMapEntry<>(14, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "E");
        assertArrayEquals(expected, map.getTable());
        assertEquals(new ExternalChainingMapEntry<>(13, "C"), map.getTable()[0].getNext());
        assertEquals(new ExternalChainingMapEntry<>(0, "A"), map.getTable()[0].getNext().getNext());

        // Testing collision + replacement
        assertEquals("A", map.put(0, "F"));
        assertEquals("B", map.put(14, "G"));
        assertEquals("C", map.put(13, "H"));
        assertEquals("D", map.put(26, "I"));
        assertEquals("E", map.put(3, "J"));
        assertEquals(13, map.getTable().length);
        assertEquals(5, map.size());
        expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "I");
        expected[1] = new ExternalChainingMapEntry<>(14, "G");
        expected[3] = new ExternalChainingMapEntry<>(3, "J");
        assertArrayEquals(expected, map.getTable());
        assertEquals(new ExternalChainingMapEntry<>(13, "H"), map.getTable()[0].getNext());
        assertEquals(new ExternalChainingMapEntry<>(0, "F"), map.getTable()[0].getNext().getNext());

        // Testing collision + replacement + resize
        map = new ExternalChainingHashMap<>();
        assertEquals(13, map.getTable().length);
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(14, "C"));
        assertNull(map.put(3, "D"));
        assertEquals("D", map.put(3, "E"));
        assertNull(map.put(27, "F"));
        assertEquals("F", map.put(27, "G"));
        assertNull(map.put(7, "H"));
        assertNull(map.put(8, "I"));
        assertNull(map.put(9, "J"));
        assertNull(map.put(24, "K"));
        assertEquals("K", map.put(24, "L"));
        assertNull(map.put(37, "M"));
        assertNull(map.put(50, "N"));
        assertNull(map.put(63, "O"));
        assertNull(map.put(16, "Q"));
        assertEquals("Q", map.put(16, "R"));
        assertEquals("R", map.put(16, "S"));
        assertEquals("O",map.put(63, "T"));
        assertNull(map.put(64, "U"));
        assertNull(map.put(2147483647, "V"));
        assertNull(map.put(-17, "W"));
        assertNull(map.put(-44, "X"));
        assertEquals("W", map.put(-17, "Y"));
        assertEquals(27, map.getTable().length);
        assertEquals(17, map.size());
        expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[27];
        expected[0] = new ExternalChainingMapEntry<>(27, "G");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "E");
        expected[7] = new ExternalChainingMapEntry<>(7, "H");
        expected[8] = new ExternalChainingMapEntry<>(8, "I");
        expected[9] = new ExternalChainingMapEntry<>(63, "T");
        expected[10] = new ExternalChainingMapEntry<>(2147483647, "V");
        expected[14] = new ExternalChainingMapEntry<>(14, "C");
        expected[16] = new ExternalChainingMapEntry<>(16, "S");
        expected[17] = new ExternalChainingMapEntry<>(-44, "X");
        expected[23] = new ExternalChainingMapEntry<>(50, "N");
        expected[24] = new ExternalChainingMapEntry<>(24, "L");
        assertArrayEquals(expected, map.getTable());
        map.put(76, "P");
        map.put(77, "P");
        assertEquals(55, map.getTable().length);
        assertEquals(19, map.size());
    }

    @Test(timeout = TIMEOUT)
    public void testPutNullKey() {
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(null, "A");
        });
    }

    @Test(timeout = TIMEOUT)
    public void testPutNullValue() {
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(0, null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testPutNullKeyValue() {
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(null, null);
        });
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemove() {
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>(6);
        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[6];
        assertArrayEquals(expected, map.getTable());
        assertEquals(0, map.size());
        assertEquals(6, map.getTable().length);

        // Remove only element
        assertNull(map.put(0, "A"));
        assertEquals("A", map.remove(0)); // should return "A"
        assertArrayEquals(expected, map.getTable());
        assertEquals(0, map.size());
        assertEquals(6, map.getTable().length);

        // Remove a few elements
        map.put(0, "B");
        map.put(1, "C");
        map.put(5, "D");
        expected[0] = new ExternalChainingMapEntry<>(0, "B");
        expected[1] = new ExternalChainingMapEntry<>(1, "C");
        expected[5] = new ExternalChainingMapEntry<>(5, "D");
        assertEquals(3, map.size());
        assertEquals("D", map.remove(5));
        expected[5] = null;
        assertEquals(2, map.size());
        assertArrayEquals(expected, map.getTable());
        assertEquals("B", map.remove(0));
        expected[0] = null;
        assertEquals(1, map.size());
        assertArrayEquals(expected, map.getTable());
        assertEquals("C", map.remove(1));
        expected[1] = null;
        assertEquals(0, map.size());
        assertArrayEquals(expected, map.getTable());
        assertEquals(6, map.getTable().length);


        // Removed from collision LL
        map.put(2, "A");
        map.put(8, "B");
        map.put(14, "C");
        map.put(20, "D");
        assertEquals(4, map.size());

        // Remove from bottom
        expected[2] = new ExternalChainingMapEntry<>(20, "D");
        assertArrayEquals(expected, map.getTable());
        assertEquals("A", map.remove(2));
        assertArrayEquals(expected, map.getTable());
        assertEquals(3, map.size());

        // Remove from middle
        assertArrayEquals(expected, map.getTable());
        assertEquals("C", map.remove(14));
        assertArrayEquals(expected, map.getTable());
        assertEquals(2, map.size());

        // Remove from top
        assertArrayEquals(expected, map.getTable());
        assertEquals("D", map.remove(20));
        expected[2] = new ExternalChainingMapEntry<>(8, "B");
        assertArrayEquals(expected, map.getTable());
        assertEquals(1, map.size());

        // Remove last element
        assertArrayEquals(expected, map.getTable());
        assertEquals("B", map.remove(8));
        expected[2] = null;
        assertArrayEquals(expected, map.getTable());
        assertEquals(0, map.size());
        assertEquals(6, map.getTable().length);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNullKey() {
        // Empty hashmap
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        assertThrows(IllegalArgumentException.class, () -> {
            map.remove(null);
        });
        // Hashmap with data
        map.put(0, "A");
        assertThrows(IllegalArgumentException.class, () -> {
            map.remove(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveKeyNotFound() {
        // Empty hashmap
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        assertThrows(NoSuchElementException.class, () -> {
            map.remove(0);
        });
        // Hashmap with data
        map.put(0, "A");
        assertThrows(NoSuchElementException.class, () -> {
            map.remove(1);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testGet() {
        // Hashmap with no collisions
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        map.put(0, "A");
        map.put(1, "B");
        map.put(5, "C");
        map.put(12, "D");
        assertEquals(4, map.size());
        assertEquals("A", map.get(0));
        assertEquals("B", map.get(1));
        assertEquals("C", map.get(5));
        assertEquals("D", map.get(12));

        // Hashmap with replacements
        assertEquals("A", map.put(0, "a"));
        assertEquals("B", map.put(1, "b"));
        assertEquals("C", map.put(5, "c"));
        assertEquals("D", map.put(12, "d"));
        assertEquals(4, map.size());
        assertEquals("a", map.get(0));
        assertEquals("b", map.get(1));
        assertEquals("c", map.get(5));
        assertEquals("d", map.get(12));

        // Hashmap with collisions
        // table[0] = <13 "E"> --> <0 "a">
        // table[1] = <40 "H"> --> <27 "G"> --> <14 "F"> --> <1 "b">
        assertNull(map.put(13, "E"));
        assertNull(map.put(14, "F"));
        assertNull(map.put(27, "G"));
        assertNull(map.put(40, "H"));
        assertEquals(8, map.size());
        assertEquals("a", map.get(0));
        assertEquals("b", map.get(1));
        assertEquals("E", map.get(13));
        assertEquals("F", map.get(14));
        assertEquals("G", map.get(27));
        assertEquals("H", map.get(40));
        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);
    }

    @Test(timeout = TIMEOUT)
    public void testGetNullKey() {
        // Empty hashmap
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        assertThrows(IllegalArgumentException.class, () -> {
            map.get(null);
        });
        // Hashmap with data
        map.put(0, "A");
        assertThrows(IllegalArgumentException.class, () -> {
            map.get(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testGetKeyNotFound() {
        // Empty hashmap
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        assertThrows(NoSuchElementException.class, () -> {
            map.get(1);
        });
        // Hashmap with data
        map.put(0, "A");
        assertThrows(NoSuchElementException.class, () -> {
            map.get(1);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testContainsKey() {
        // Hashmap with no collisions
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        assertFalse(map.containsKey(0));
        map.put(0, "A");
        map.put(1, "B");
        map.put(5, "C");
        map.put(12, "D");
        assertEquals(4, map.size());
        assertTrue(map.containsKey(0));
        assertTrue(map.containsKey(1));
        assertTrue(map.containsKey(5));
        assertTrue(map.containsKey(12));
        assertFalse(map.containsKey(-1));
        assertFalse(map.containsKey(100));
        assertFalse(map.containsKey(28));
        assertFalse(map.containsKey(53));

        // Hashmap with replacements
        assertEquals("A", map.put(0, "a"));
        assertEquals("B", map.put(1, "b"));
        assertEquals("C", map.put(5, "c"));
        assertEquals("D", map.put(12, "d"));
        assertEquals(4, map.size());
        assertTrue(map.containsKey(0));
        assertTrue(map.containsKey(1));
        assertTrue(map.containsKey(5));
        assertTrue(map.containsKey(12));
        assertFalse(map.containsKey(-1));
        assertFalse(map.containsKey(100));
        assertFalse(map.containsKey(28));
        assertFalse(map.containsKey(53));

        // Hashmap with collisions
        // table[0] = <13 "E"> --> <0 "a">
        // table[1] = <40 "H"> --> <27 "G"> --> <14 "F"> --> <1 "b">
        assertNull(map.put(13, "E"));
        assertNull(map.put(14, "F"));
        assertNull(map.put(27, "G"));
        assertNull(map.put(40, "H"));
        assertEquals(8, map.size());
        assertTrue(map.containsKey(13));
        assertTrue(map.containsKey(0));
        assertTrue(map.containsKey(40));
        assertTrue(map.containsKey(27));
        assertTrue(map.containsKey(14));
        assertTrue(map.containsKey(1));
        assertFalse(map.containsKey(-1));
        assertFalse(map.containsKey(100));
        assertFalse(map.containsKey(28));
        assertFalse(map.containsKey(53));
        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        // Removing
        assertEquals("H", map.remove(40));
        assertFalse(map.containsKey(40));
        assertEquals("E", map.remove(13));
        assertFalse(map.containsKey(13));
        assertEquals(6, map.size());
        assertEquals(13, map.getTable().length);
    }

    @Test(timeout = TIMEOUT)
    public void testContainsKeyNull() {
        // Empty hashmap
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        assertThrows(IllegalArgumentException.class, () -> {
            map.containsKey(null);
        });
        // Hashmap with data
        map.put(0, "A");
        assertThrows(IllegalArgumentException.class, () -> {
            map.containsKey(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testKeySet() {
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        Set<Integer> expected = new HashSet<>();
        assertEquals(expected, map.keySet());

        // Hashmap with no collisions
        map.put(0, "A");
        map.put(1, "B");
        map.put(5, "C");
        map.put(12, "D");
        assertEquals(4, map.size());
        expected.add(0);
        expected.add(1);
        expected.add(5);
        expected.add(12);
        assertEquals(expected, map.keySet());

        // Hashmap with replacements
        assertEquals("A", map.put(0, "a"));
        assertEquals("B", map.put(1, "b"));
        assertEquals("C", map.put(5, "c"));
        assertEquals("D", map.put(12, "d"));
        assertEquals(4, map.size());
        assertEquals(expected, map.keySet());

        // Hashmap with collisions
        // table[0] = <13 "E"> --> <0 "a">
        // table[1] = <40 "H"> --> <27 "G"> --> <14 "F"> --> <1 "b">
        assertNull(map.put(13, "E"));
        assertNull(map.put(14, "F"));
        assertNull(map.put(27, "G"));
        assertNull(map.put(40, "H"));
        assertEquals(8, map.size());
        expected.add(13);
        expected.add(14);
        expected.add(27);
        expected.add(40);
        assertEquals(expected, map.keySet());

        // Removing
        assertEquals("H", map.remove(40));
        expected.remove(40);
        assertEquals(expected, map.keySet());
        assertEquals("E", map.remove(13));
        expected.remove(13);
        assertEquals(expected, map.keySet());
        assertEquals(6, map.size());
        assertEquals(13, map.getTable().length);
    }

    @Test(timeout = TIMEOUT)
    public void testValues() {
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        LinkedList<String> expected = new LinkedList<>();
        assertEquals(expected, map.values());

        // Hashmap with no collisions
        map.put(0, "A");
        map.put(1, "B");
        map.put(5, "C");
        map.put(12, "C");
        assertEquals(4, map.size());
        expected.add("A");
        expected.add("B");
        expected.add("C");
        expected.add("C");
        assertEquals(expected, map.values());

        // Hashmap with replacements
        assertEquals("A", map.put(0, "a"));
        assertEquals("B", map.put(1, "b"));
        assertEquals("C", map.put(5, "b"));
        assertEquals("C", map.put(12, "d"));
        assertEquals(4, map.size());
        expected.clear();
        expected.add("a");
        expected.add("b");
        expected.add("b");
        expected.add("d");
        assertEquals(expected, map.values());

        // Hashmap with collisions
        // table[0] = <13 "E"> --> <0 "a">
        // table[1] = <40 "H"> --> <27 "G"> --> <14 "F"> --> <1 "b">
        assertNull(map.put(13, "E"));
        assertNull(map.put(14, "F"));
        assertNull(map.put(27, "d"));
        assertNull(map.put(40, "H"));
        assertEquals(8, map.size());
        expected.clear();
        expected.add("E");
        expected.add("a");
        expected.add("H");
        expected.add("d");
        expected.add("F");
        expected.add("b");
        expected.add("b");
        expected.add("d");
        assertEquals(expected, map.values());

        // Removing
        assertEquals("H", map.remove(40));
        expected.remove("H");
        assertEquals(expected, map.values());
        assertEquals("E", map.remove(13));
        expected.remove("E");
        assertEquals(expected, map.values());
        assertEquals(6, map.size());
        assertEquals(13, map.getTable().length);
    }

    @Test(timeout = TIMEOUT)
    public void testResize() {
        // Small hashmap resize 4 --> 9
        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[4];
        map = new ExternalChainingHashMap<>(4);     // [null, null, null, null]
        assertArrayEquals(expected, map.getTable());
        assertNull(map.put(4, "A"));                       // [<4 "A">, null, null, null]
        expected[0] = new ExternalChainingMapEntry<>(4, "A");
        assertArrayEquals(expected, map.getTable());
        assertNull(map.put(7, "D"));                       // [<4 "A">, null, null, <7 "D">]
        expected[3] = new ExternalChainingMapEntry<>(7, "D");
        assertArrayEquals(expected, map.getTable());
        assertEquals(4, map.getTable().length);
        assertNull(map.put(5, "B"));                       // RESIZE HERE --> [null, null, null, null, <4 "A">, <5 "B">, null, <7 "D">, null]
        assertEquals(9, map.getTable().length);
        expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[9];
        expected[4] = new ExternalChainingMapEntry<>(4, "A");
        expected[5] = new ExternalChainingMapEntry<>(5, "B");
        expected[7] = new ExternalChainingMapEntry<>(7, "D");
        assertArrayEquals(expected, map.getTable());
        assertNull(map.put(6, "C"));                       // [null, null, null, null, <4 "A">, <5 "B">, <6 "C">, <7 "D">, null]
        expected[6] = new ExternalChainingMapEntry<>(6, "C");
        assertArrayEquals(expected, map.getTable());

        // Resize again 9 --> 19
        assertNull(map.put(0, "E"));                       // [<0 "E">, null, null, null, <4 "A">, <5 "B">, <6 "C">, <7 "D">, null]
        expected[0] = new ExternalChainingMapEntry<>(0, "E");
        assertArrayEquals(expected, map.getTable());
        assertNull(map.put(10, "F"));                       // [<0 "E">, <10 "F">, null, null, <4 "A">, <5 "B">, <6 "C">, <7 "D">, null]
        expected[1] = new ExternalChainingMapEntry<>(10, "F");
        assertArrayEquals(expected, map.getTable());
        assertEquals(9, map.getTable().length);
        assertNull(map.put(21, "G"));                       // RESIZE HERE --> [<0 "E">, null, <21 "G">, null, <4 "A">, <5 "B">, <6 "C">, <7 "D">, null, null, <10 "F">, null, null, null, null, null, null, null, null]
        assertEquals(19, map.getTable().length);
        expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[19];
        expected[0] = new ExternalChainingMapEntry<>(0, "E");
        expected[2] = new ExternalChainingMapEntry<>(21, "G");
        expected[4] = new ExternalChainingMapEntry<>(4, "A");
        expected[5] = new ExternalChainingMapEntry<>(5, "B");
        expected[6] = new ExternalChainingMapEntry<>(6, "C");
        expected[7] = new ExternalChainingMapEntry<>(7, "D");
        expected[10] = new ExternalChainingMapEntry<>(10, "F");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testClear() {
        ExternalChainingHashMap<Integer, String> map = new ExternalChainingHashMap<>();
        assertNull(map.put(0, "B"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(152, "B"));
        assertNull(map.put(3, "B"));
        assertNull(map.put(224, "B"));
        assertNull(map.put(105, "B"));
        assertNull(map.put(66, "B"));
        assertNull(map.put(57, "B"));
        assertNull(map.put(38, "B"));
        assertNull(map.put(19, "B"));
        assertEquals(10, map.size());

        map.clear();
        assertEquals(0, map.size());
        assertArrayEquals(new ExternalChainingMapEntry[ExternalChainingHashMap.INITIAL_CAPACITY], map.getTable());
    }
}
