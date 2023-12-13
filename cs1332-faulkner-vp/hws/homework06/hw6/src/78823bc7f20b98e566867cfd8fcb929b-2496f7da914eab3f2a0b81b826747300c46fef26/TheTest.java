import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

/**
 * Test Cases for HW6
 * @author Akhil Kothapalli
 * @version 1.0
 */
public class TheTest {

    private static final int TIMEOUT = 200;
    private ExternalChainingHashMap<Integer, String> map;

    @Before
    public void setUp() {
        map = new ExternalChainingHashMap<>(10);
    }

    @Test(timeout = TIMEOUT)
    public void testInitialization() {
        assertEquals(0, map.size());
        assertArrayEquals(new ExternalChainingMapEntry[10], map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testPut() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(5, "E");

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[0] = null;
        expected[1] = new ExternalChainingMapEntry<>(1, "A");
        expected[2] = new ExternalChainingMapEntry<>(2, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "C");
        expected[4] = new ExternalChainingMapEntry<>(4, "D");
        expected[5] = new ExternalChainingMapEntry<>(5, "E");
        assertArrayEquals(expected, map.getTable());
    }
    @Test(timeout = TIMEOUT)
    public void testPutWithSameKey() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(5, "E");
        map.put(1, "F");

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[0] = null;
        expected[1] = new ExternalChainingMapEntry<>(1, "F");
        expected[2] = new ExternalChainingMapEntry<>(2, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "C");
        expected[4] = new ExternalChainingMapEntry<>(4, "D");
        expected[5] = new ExternalChainingMapEntry<>(5, "E");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testPutWithCollision() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(5, "E");
        map.put(11, "F");

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[0] = null;
        expected[1] = new ExternalChainingMapEntry<>(11, "F", new ExternalChainingMapEntry<>(1, "A"));
        expected[2] = new ExternalChainingMapEntry<>(2, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "C");
        expected[4] = new ExternalChainingMapEntry<>(4, "D");
        expected[5] = new ExternalChainingMapEntry<>(5, "E");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testPutWithResizeAndLinkedData() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(5, "E");
        map.put(11, "K");
        map.put(6, "F");
        map.put(7, "G");
        map.put(8, "H");

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[21];
        expected[0] = null;
        expected[1] = new ExternalChainingMapEntry<>(1, "A");
        expected[2] = new ExternalChainingMapEntry<>(2, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "C");
        expected[4] = new ExternalChainingMapEntry<>(4, "D");
        expected[5] = new ExternalChainingMapEntry<>(5, "E");
        expected[6] = new ExternalChainingMapEntry<>(6, "F");
        expected[7] = new ExternalChainingMapEntry<>(7, "G");
        expected[8] = new ExternalChainingMapEntry<>(8, "H");
        expected[11] = new ExternalChainingMapEntry<>(11, "K");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testPutExceptionKey() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(null, "Dog");
        });
    }

    @Test(timeout = TIMEOUT)
    public void testPutExceptionValue() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(1, null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRemove() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(5, "E");
        assertEquals(5, map.size());
        assertEquals("A", map.remove(1));
        assertEquals("C", map.remove(3));

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[0] = null;
        expected[2] = new ExternalChainingMapEntry<>(2, "B");
        expected[4] = new ExternalChainingMapEntry<>(4, "D");
        expected[5] = new ExternalChainingMapEntry<>(5, "E");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveFromLinkedSegment00() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(5, "E");
        map.put(11, "K");

        assertEquals(6, map.size());

        assertEquals("K", map.remove(11));

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[0] = null;
        expected[1] = new ExternalChainingMapEntry<>(1, "A");
        expected[2] = new ExternalChainingMapEntry<>(2, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "C");
        expected[4] = new ExternalChainingMapEntry<>(4, "D");
        expected[5] = new ExternalChainingMapEntry<>(5, "E");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveFromLinkedSegment01() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(5, "E");
        map.put(11, "K");

        assertEquals(6, map.size());

        assertEquals("A", map.remove(1));

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[0] = null;
        expected[1] = new ExternalChainingMapEntry<>(11, "K");
        expected[2] = new ExternalChainingMapEntry<>(2, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "C");
        expected[4] = new ExternalChainingMapEntry<>(4, "D");
        expected[5] = new ExternalChainingMapEntry<>(5, "E");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveFromLinkedSegment02() {
        map.put(1, "A");
        map.put(11, "B");
        map.put(21, "C");
        map.put(31, "D");

        assertEquals(4, map.size());

        assertEquals("C", map.remove(21));

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[0] = null;
        expected[1] = new ExternalChainingMapEntry<>(31, "D",
                new ExternalChainingMapEntry<>(11, "B",
                        new ExternalChainingMapEntry<>(1, "A")));
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveExceptionIllegal() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.remove(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveExceptionNoSuch() {
        assertThrows(NoSuchElementException.class, () -> {
            map.put(1, "A");
            map.remove(3);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testGet() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(5, "E");
        assertEquals(5, map.size());

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[1] = new ExternalChainingMapEntry<>(1, "A");
        expected[2] = new ExternalChainingMapEntry<>(2, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "C");
        expected[4] = new ExternalChainingMapEntry<>(4, "D");
        expected[5] = new ExternalChainingMapEntry<>(5, "E");
        assertArrayEquals(expected, map.getTable());

        assertEquals("C", map.get(3));
        assertEquals("E", map.get(5));
    }

    @Test(timeout = TIMEOUT)
    public void testGetWithLinkedSegment() {
        map.put(1, "A");
        map.put(11, "B");
        map.put(21, "C");
        map.put(31, "D");

        assertEquals(4, map.size());
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[1] = new ExternalChainingMapEntry<>(31, "D", new ExternalChainingMapEntry<>(21, "C", new ExternalChainingMapEntry<>(11, "B", new ExternalChainingMapEntry<>(1, "A"))));
        assertArrayEquals(expected, map.getTable());
        assertEquals("C", map.get(21));
    }

    @Test(timeout = TIMEOUT)
    public void testGetExceptionIllegal() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.get(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testGetExceptionNoSuch() {
        assertThrows(NoSuchElementException.class, () -> {
            map.put(1, "A");
            map.get(3);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testContains() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(5, "E");
        assertEquals(5, map.size());

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[1] = new ExternalChainingMapEntry<>(1, "A");
        expected[2] = new ExternalChainingMapEntry<>(2, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "C");
        expected[4] = new ExternalChainingMapEntry<>(4, "D");
        expected[5] = new ExternalChainingMapEntry<>(5, "E");
        assertArrayEquals(expected, map.getTable());

        assertEquals(true, map.containsKey(3));
        assertEquals(false, map.containsKey(21));
    }

    @Test(timeout = TIMEOUT)
    public void testContainsWithLinkedSegments() {
        map.put(1, "A");
        map.put(11, "B");
        map.put(21, "C");
        map.put(31, "D");

        assertEquals(4, map.size());
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[1] = new ExternalChainingMapEntry<>(31, "D", new ExternalChainingMapEntry<>(21, "C", new ExternalChainingMapEntry<>(11, "B", new ExternalChainingMapEntry<>(1, "A"))));
        assertEquals(true, map.containsKey(31));
        assertEquals(true, map.containsKey(21));
        assertEquals(false, map.containsKey(41));
    }

    @Test(timeout = TIMEOUT)
    public void testContainsIllegal() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.containsKey(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testValuesAfterResize() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(5, "E");
        map.put(11, "K");
        map.put(6, "F");
        map.put(7, "G");
        map.put(8, "H");

        assertEquals(9, map.size());

        List<String> expec = new ArrayList<>();
        expec.add("A");
        expec.add("B");
        expec.add("C");
        expec.add("D");
        expec.add("E");
        expec.add("F");
        expec.add("G");
        expec.add("H");
        expec.add("K");
        assertEquals(expec, map.values());
    }

    @Test(timeout = TIMEOUT)
    public void testValuesWithLinkedSegments() {
        map.put(1, "A");
        map.put(11, "B");
        map.put(21, "C");
        map.put(31, "D");

        assertEquals(4, map.size());

        List<String> expec = new ArrayList<>();
        expec.add("D");
        expec.add("C");
        expec.add("B");
        expec.add("A");
        assertEquals(expec, map.values());
    }

    @Test(timeout = TIMEOUT)
    public void testKeySetAfterResize() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(5, "E");
        map.put(11, "K");
        map.put(6, "F");
        map.put(7, "G");
        map.put(8, "H");

        assertEquals(9, map.size());

        HashSet<Integer> expec = new HashSet<>();
        expec.add(1);
        expec.add(2);
        expec.add(3);
        expec.add(4);
        expec.add(5);
        expec.add(6);
        expec.add(7);
        expec.add(8);
        expec.add(11);
        assertEquals(expec, map.keySet());
    }

    @Test(timeout = TIMEOUT)
    public void testKeySetWithLinkedSegments() {
        map.put(1, "A");
        map.put(11, "B");
        map.put(21, "C");
        map.put(31, "D");

        assertEquals(4, map.size());

        HashSet<Integer> expec = new HashSet<>();
        expec.add(1);
        expec.add(11);
        expec.add(21);
        expec.add(31);
        assertEquals(expec, map.keySet());
    }

    @Test(timeout = TIMEOUT)
    public void testResize() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(11, "K");

        map.resizeBackingTable(8);
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[8];
        expected[1] = new ExternalChainingMapEntry<>(1, "A");
        expected[2] = new ExternalChainingMapEntry<>(2, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "C", new ExternalChainingMapEntry<>(11, "K"));
        expected[4] = new ExternalChainingMapEntry<>(4, "D");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testResizeExceptionIllegal() {
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(1, "A");
            map.put(2, "B");
            map.put(3, "C");
            map.put(4, "D");
            map.put(11, "K");
            map.resizeBackingTable(2);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testClear() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(11, "K");

        assertEquals(5, map.size());

        map.clear();
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[13];
        assertArrayEquals(expected, map.getTable());
        assertEquals(0, map.size());
    }
}
