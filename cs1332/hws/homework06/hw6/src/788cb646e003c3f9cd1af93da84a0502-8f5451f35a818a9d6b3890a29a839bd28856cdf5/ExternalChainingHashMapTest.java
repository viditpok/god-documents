import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

/**
 * @author Ahaan Limaye
 * @version 1.0
 */
public class ExternalChainingHashMapTest {
    private static final int TIMEOUT = 200;
    private ExternalChainingHashMap<Integer, String> map;

    @Before
    public void setUp() { map = new ExternalChainingHashMap<>(); };

    @Test(timeout = TIMEOUT)
    public void testInitialization() {
        assertEquals(0, map.size());
        assertArrayEquals(new ExternalChainingHashMap[ExternalChainingHashMap.INITIAL_CAPACITY], map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testInitializationWithCapacity() {
        map = new ExternalChainingHashMap<>(20);
        assertEquals(0, map.size());
        assertArrayEquals(new ExternalChainingHashMap[20], map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testPut() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testPutDuplicate() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertEquals("E", map.put(14, "A"));
        assertEquals("A", map.put(0, "B"));

        assertEquals(6, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "B"));
        expected[1] = new ExternalChainingMapEntry<>(14, "A", new ExternalChainingMapEntry<>(1, "B"));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testPutResize() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));
        assertNull(map.put(28, "I"));

        assertEquals(9, map.size());
        assertEquals(27, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[27];
        expected[0] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(0, "A"));
        expected[1] = new ExternalChainingMapEntry<>(28, "I", new ExternalChainingMapEntry<>(1, "B"));
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[13] = new ExternalChainingMapEntry<>(13, "D");
        expected[14] = new ExternalChainingMapEntry<>(14, "E");
        expected[15] = new ExternalChainingMapEntry<>(15, "F");
        expected[26] = new ExternalChainingMapEntry<>(26, "G");

        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testPutNull1() {
        map.put(null, "A");
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testPutNull2() {
        map.put(0, null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testPutNull3() {
        map.put(null, null);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveFrontOfChain() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());

        assertEquals("G", map.remove((26)));
        assertEquals("H", map.remove((27)));

        assertEquals(6, map.size());
        assertEquals(13, map.getTable().length);

        expected[0] = new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A"));
        expected[1] = new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B"));

        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveMiddleOfChain() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());

        assertEquals("D", map.remove((13)));
        assertEquals("E", map.remove((14)));

        assertEquals(6, map.size());
        assertEquals(13, map.getTable().length);

        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(0, "A"));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(1, "B"));

        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveEndOfChain() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());

        assertEquals("A", map.remove((0)));
        assertEquals("B", map.remove((1)));

        assertEquals(6, map.size());
        assertEquals(13, map.getTable().length);

        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D"));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E"));

        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testRemoveNotInMap() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());

        map.remove(39);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testRemoveNull() {
        assertNull(map.put(0, "A"));
        map.remove(null);
    }

    @Test(timeout = TIMEOUT)
    public void testGet() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());

        assertEquals("E", map.get(14));
        assertEquals("G", map.get(26));
        assertEquals("A", map.get(0));
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testGetNotInMap() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());

        map.get(39);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testGetNull() {
        assertNull(map.put(0, "A"));
        map.get(null);
    }

    @Test(timeout = TIMEOUT)
    public void testContainsKey() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());

        assertTrue(map.containsKey(14));
        assertTrue(map.containsKey(26));
        assertTrue(!map.containsKey(39));
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testContainsKeyNull() {
        assertNull(map.put(0, "A"));
        map.containsKey(null);
    }

    @Test(timeout = TIMEOUT)
    public void testKeySet() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());

        Set<Integer> expectedSet = new HashSet<>();
        expectedSet.add(0);
        expectedSet.add(1);
        expectedSet.add(2);
        expectedSet.add(13);
        expectedSet.add(14);
        expectedSet.add(15);
        expectedSet.add(26);
        expectedSet.add(27);

        assertEquals(expectedSet, map.keySet());
    }

    @Test(timeout = TIMEOUT)
    public void testValues() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());

        List<String> expectedList = new ArrayList<>();
        expectedList.add("G");
        expectedList.add("D");
        expectedList.add("A");
        expectedList.add("H");
        expectedList.add("E");
        expectedList.add("B");
        expectedList.add("F");
        expectedList.add("C");

        assertEquals(expectedList, map.values());
    }

    @Test(timeout = TIMEOUT)
    public void testResize() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());

        map.resizeBackingTable(8);

        expected = new ExternalChainingMapEntry[8];

        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C", new ExternalChainingMapEntry<>(26, "G"));
        expected[3] = new ExternalChainingMapEntry<>(27, "H");
        expected[5] = new ExternalChainingMapEntry<>(13, "D");
        expected[6] = new ExternalChainingMapEntry<>(14, "E");
        expected[7] = new ExternalChainingMapEntry<>(15, "F");

        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testResizeLengthLessThanSize() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(26, "G", new ExternalChainingMapEntry<>(13, "D", new ExternalChainingMapEntry<>(0, "A")));
        expected[1] = new ExternalChainingMapEntry<>(27, "H", new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(15, "F", new ExternalChainingMapEntry<>(2, "C"));

        assertArrayEquals(expected, map.getTable());

        map.resizeBackingTable(2);
    }

    @Test(timeout = TIMEOUT)
    public void testClear() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(13, "D"));
        assertNull(map.put(14, "E"));
        assertNull(map.put(15, "F"));
        assertNull(map.put(26, "G"));
        assertNull(map.put(27, "H"));
        assertNull(map.put(28, "I"));

        assertEquals(9, map.size());
        assertEquals(27, map.getTable().length);

        map.clear();

        ExternalChainingMapEntry<Integer, String>[] expected = new ExternalChainingMapEntry[13];

        assertArrayEquals(expected, map.getTable());
        assertEquals(0, map.size());
        assertEquals(13, map.getTable().length);
    }
}