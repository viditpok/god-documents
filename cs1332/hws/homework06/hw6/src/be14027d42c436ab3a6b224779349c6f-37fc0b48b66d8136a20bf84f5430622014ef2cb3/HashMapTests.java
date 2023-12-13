import org.junit.Before;
import org.junit.Test;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;

import static org.junit.Assert.*;

/**
 * @author Rishi Soni
 * @version 1.0
 */
public class HashMapTests {

    private static final int TIMEOUT = 200;
    private ExternalChainingHashMap<Integer, String> map;

    @Before
    public void setUp() {
        map = new ExternalChainingHashMap<>();
    }

    @Test(timeout = TIMEOUT)
    public void testInitialization() {
        assertEquals(0, map.size());
        assertArrayEquals(new ExternalChainingMapEntry[ExternalChainingHashMap
                .INITIAL_CAPACITY], map.getTable());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testExceptions() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));

        //Put Method
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(null, "A");
        });
        assertThrows(IllegalArgumentException.class, () -> {
            map.put(1, null);
        });

        //Remove Method
        assertThrows(NoSuchElementException.class, () -> {
            map.remove(7);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            map.remove(null);
        });

        //Get Method
        assertThrows(IllegalArgumentException.class, () -> {
            map.get(null);
        });
        assertThrows(NoSuchElementException.class, () -> {
            map.get(7);
        });

        //ContainsKey Method
        assertThrows(IllegalArgumentException.class, () -> {
            map.containsKey(null);
        });

        //ResizeBackingTable Method
        assertThrows(IllegalArgumentException.class, () -> {
            map.resizeBackingTable(3);
        });
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPut() {

        //Standard Cases
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));
        assertNull(map.put(13, "H"));

        assertEquals(8, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[ExternalChainingHashMap
                                .INITIAL_CAPACITY];
        expected[0] = new ExternalChainingMapEntry<>(13, "H", new ExternalChainingMapEntry<>(0, "A"));
        assertEquals(expected[0].getNext(), map.getTable()[0].getNext());
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");
        expected[5] = new ExternalChainingMapEntry<>(5, "F");
        expected[6] = new ExternalChainingMapEntry<>(6, "G");
        assertArrayEquals(expected, map.getTable());

        //Resize case
        assertNull(map.put(7, "H"));
        assertNull(map.put(8, "I"));
        assertEquals(10, map.size());
        assertEquals(27, map.getTable().length);

        expected = (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[27];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");
        expected[5] = new ExternalChainingMapEntry<>(5, "F");
        expected[6] = new ExternalChainingMapEntry<>(6, "G");
        expected[7] = new ExternalChainingMapEntry<>(7, "H");
        expected[8] = new ExternalChainingMapEntry<>(8, "I");
        expected[13] = new ExternalChainingMapEntry<>(13, "H");
        assertArrayEquals(expected, map.getTable());

        //Duplicate
        assertEquals("G", map.put(6, "Z"));
        expected[6] = new ExternalChainingMapEntry<>(6, "Z");
        assertEquals(10, map.size()); //Size shouldn't change
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testRemove() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));
        assertNull(map.put(13, "H"));

        assertEquals("H", map.remove(13));
        assertEquals("E", map.remove(4));
        assertEquals("A", map.remove(0));

        assertEquals(5, map.size());
        assertEquals(13, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[ExternalChainingHashMap
                                .INITIAL_CAPACITY];
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[5] = new ExternalChainingMapEntry<>(5, "F");
        expected[6] = new ExternalChainingMapEntry<>(6, "G");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testGet() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));
        assertNull(map.put(13, "H"));

        assertEquals("A", map.get(0));
        assertEquals("E", map.get(4));
        assertEquals("G", map.get(6));
    }
    @Test(timeout = TIMEOUT)
    public void testContainsKey() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));
        assertNull(map.put(13, "H"));

        assertEquals(true, map.containsKey(0));
        assertEquals(false, map.containsKey(14));
        assertEquals(true, map.containsKey(6));
    }

    @Test(timeout = TIMEOUT)
    public void testKeySet() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));
        assertNull(map.put(13, "H"));

        Set<Integer> expected = new HashSet<>();
        expected.add(13);
        expected.add(0);
        expected.add(1);
        expected.add(2);
        expected.add(3);
        expected.add(4);
        expected.add(5);
        expected.add(6);
        assertEquals(expected, map.keySet());
    }

    @Test(timeout = TIMEOUT)
    public void testValues() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));
        assertNull(map.put(13, "H"));

        List<String> expected = new LinkedList<>();
        expected.add("H");
        expected.add("A");
        expected.add("B");
        expected.add("C");
        expected.add("D");
        expected.add("E");
        expected.add("F");
        expected.add("G");
        assertEquals(expected, map.values());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testResize() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));
        assertNull(map.put(19, "H"));

        map.resizeBackingTable(10);

        assertEquals(8, map.size());
        assertEquals(10, map.getTable().length);

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[10];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");
        expected[5] = new ExternalChainingMapEntry<>(5, "F");
        expected[6] = new ExternalChainingMapEntry<>(6, "G");
        expected[9] = new ExternalChainingMapEntry<>(19, "H");
        assertArrayEquals(expected, map.getTable());

        map.resizeBackingTable(8);

        assertEquals(8, map.size());
        assertEquals(8, map.getTable().length);

        expected = (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[8];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(19, "H");
        assertEquals(new ExternalChainingMapEntry<>(3, "D"), map.getTable()[3].getNext());
        expected[4] = new ExternalChainingMapEntry<>(4, "E");
        expected[5] = new ExternalChainingMapEntry<>(5, "F");
        expected[6] = new ExternalChainingMapEntry<>(6, "G");
        assertArrayEquals(expected, map.getTable());

    }

    @Test(timeout = TIMEOUT)
    public void testClear() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));
        assertNull(map.put(13, "H"));

        map.clear();
        assertEquals(0, map.size());
        assertArrayEquals(new ExternalChainingMapEntry[ExternalChainingHashMap
                .INITIAL_CAPACITY], map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void everything() {
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(-2, "C")); //Hash function should account for negatives
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));
        assertNull(map.put(15, "H"));
        assertNull(map.put(28, "I"));

        assertEquals(true, map.containsKey(15));
        assertEquals(true, map.containsKey(-2));

        assertEquals("H", map.get(15));
        assertEquals("I", map.get(28));

        assertEquals("I", map.remove(28));
        assertEquals("C", map.remove(-2));
        assertEquals("H", map.remove(15));

        assertEquals(false, map.containsKey(-3));
        assertEquals(false, map.containsKey(28));

        assertEquals(27, map.getTable().length);
        assertEquals(6, map.size());

        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[27];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");
        expected[5] = new ExternalChainingMapEntry<>(5, "F");
        expected[6] = new ExternalChainingMapEntry<>(6, "G");
        assertArrayEquals(expected, map.getTable());

        map.resizeBackingTable(17);

        assertNull(map.put(2, "1"));
        assertNull(map.put(19, "2"));
        assertNull(map.put(36, "3"));
        assertEquals("3", map.put(36, "Nope")); //Doesn't affect size
        assertNull(map.put(53, "4"));
        assertEquals("2", map.put(19, "5")); //Doesn't affect size

        assertEquals(17, map.getTable().length);
        assertEquals(10, map.size());

        expected = (ExternalChainingMapEntry<Integer, String>[])
                        new ExternalChainingMapEntry[17];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(53, "4");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");
        expected[5] = new ExternalChainingMapEntry<>(5, "F");
        expected[6] = new ExternalChainingMapEntry<>(6, "G");
        assertArrayEquals(expected, map.getTable());

        assertEquals(new ExternalChainingMapEntry<>(36, "Nope"), map.getTable()[2].getNext());
        assertEquals(new ExternalChainingMapEntry<>(19, "5"), map.getTable()[2].getNext().getNext());
        assertEquals(new ExternalChainingMapEntry<>(2, "1"), map.getTable()[2].getNext().getNext().getNext());

        assertEquals("5", map.remove(19));
        assertEquals("Nope", map.remove(36));
        assertEquals("4", map.remove(53));
        
        assertEquals(new ExternalChainingMapEntry<>(2, "1"), map.getTable()[2]);
        assertEquals(7, map.size());
    }
}
