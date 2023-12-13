import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;

import static org.junit.Assert.*;

/**
 *  Set of tests that I created to try and test different edge cases for HashMaps
 *
 * @author Abhinav Vemulapalli
 * @version 1.0
 */
public class HashMapTest {

    private static final int TIMEOUT = 200;
    private ExternalChainingHashMap<Integer, String> map;
    private ExternalChainingHashMap<String, String> stringMap;

    @Before
    public void setUp() {
        map = new ExternalChainingHashMap<>();
        stringMap = new ExternalChainingHashMap<>();
    }

    @Test(timeout = TIMEOUT)
    public void testResizeOnDuplicateButGreaterLF() {
        for (int i = 0; i < 8; i++) {
            assertNull(map.put(i, Integer.toString(i)));
        }
        assertEquals(8, map.size());

        assertEquals("0", map.put(0, "RESIZE"));
        assertEquals(8, map.size());

        ExternalChainingMapEntry<Integer, String> expected[] =
                (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[27];

        expected[0] = new ExternalChainingMapEntry<>(0, "RESIZE");
        for (int i = 1; i < 8; i++) {
            expected[i] = new ExternalChainingMapEntry<>(i, Integer.toString(i));
        }
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testEqualsUsage() {
        String temp = "bacon";
        assertNull(map.put(10, temp));
        assertEquals(1, map.size());

        assertEquals(temp, map.get(new Integer(10)));

        String temp2 = "ice cream";
        assertEquals(temp, map.put(new Integer(10), temp2));
        assertEquals(1,  map.size());

        assertEquals(temp2, map.get(new Integer(10)));

        assertTrue(map.containsKey(new Integer(10)));

        assertEquals(temp2, map.remove(new Integer(10)));
    }

    @Test(timeout = TIMEOUT)
    public void testHashCodeCollision() {
        assertNull(stringMap.put("Aa", "0"));
        assertNull(stringMap.put("BB", "1"));
        assertEquals(2, stringMap.size());

        ExternalChainingMapEntry<String, String> expected[] =
                (ExternalChainingMapEntry<String, String>[]) new ExternalChainingMapEntry[13];

        ExternalChainingMapEntry<String, String> temp = new ExternalChainingMapEntry<>("Aa", "0");
        // hashcode of "Aa" is 2112. 2112 % 13 = 6
        expected[6] = new ExternalChainingMapEntry<>("BB", "1", temp);

        assertArrayEquals(expected, stringMap.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testExternalChainingFunctionality() {
        assertNull(map.put(0, "1"));
        assertNull(map.put(13, "2"));
        assertEquals(2, map.size());

        ExternalChainingMapEntry<Integer, String> expected[] =
                (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];

        ExternalChainingMapEntry<Integer, String> temp = new ExternalChainingMapEntry<>(0, "1");
        expected[0] = new ExternalChainingMapEntry<>(13, "2", temp);

        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testExternalChainingGet() {
        assertNull(stringMap.put("Aa", "0"));
        assertNull(stringMap.put("BB", "0.1"));
        assertNull(stringMap.put("AA", "2"));
        assertEquals(3, stringMap.size());

        assertNull(map.put(0, "1"));
        assertNull(map.put(13, "1.2"));
        assertNull(map.put(26, "1.2.3"));
        assertNull(map.put(1, "2"));
        assertEquals(4, map.size());

        assertEquals("0", stringMap.get("Aa"));
        assertEquals("0.1", stringMap.get("BB"));
        assertEquals("2", stringMap.get("AA"));
        assertEquals("1", map.get(0));
        assertEquals("1.2", map.get(13));
        assertEquals("1.2.3", map.get(26));
        assertEquals("2", map.get(1));
    }

    @Test(timeout = TIMEOUT)
    public void testExternalChainingRemove() {
        String one = "1";
        String oneTwo = "1.2";
        String oneTwoThree = "1.2.3";
        String oneTwoThreeFour = "1.2.3.4";
        String oneTwoThreeFourFive = "1.2.3.4.5";
        String zero = "0";
        String zeroOne = "0.1";

        assertNull(map.put(0, one));
        assertNull(map.put(13, oneTwo));
        assertNull(map.put(26, oneTwoThree));
        assertNull(map.put(39, oneTwoThreeFour));
        assertNull(map.put(52, oneTwoThreeFourFive));
        assertEquals(5, map.size());

        assertNull(stringMap.put("Aa", zero));
        assertNull(stringMap.put("BB", zeroOne));
        assertEquals(2, stringMap.size());

        assertEquals(oneTwo, map.remove(13));
        assertEquals(4, map.size());
        ExternalChainingMapEntry<Integer, String> expected[] =
                (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];
        ExternalChainingMapEntry<Integer, String> node0 = new ExternalChainingMapEntry<>(0, one);
        ExternalChainingMapEntry<Integer, String> node26 = new ExternalChainingMapEntry<>(26, oneTwoThree, node0);
        ExternalChainingMapEntry<Integer, String> node39 = new ExternalChainingMapEntry<>(39, oneTwoThreeFour, node26);
        ExternalChainingMapEntry<Integer, String> node52 = new ExternalChainingMapEntry<>(52, oneTwoThreeFourFive, node39);
        expected[0] = node52 ;
        assertArrayEquals(expected, map.getTable());

        assertEquals(oneTwoThreeFourFive, map.remove(52));
        assertEquals(3, map.size());
        expected[0] = node39;
        assertArrayEquals(expected, map.getTable());

        assertEquals(oneTwoThree, map.remove(26));
        assertEquals(2, map.size());
        node39.setNext(node0);
        expected[0] = node39;
        assertArrayEquals(expected, map.getTable());

        assertEquals(one, map.remove(0));
        assertEquals(1, map.size());
        node39.setNext(null);
        expected[0] = node39;
        assertArrayEquals(expected, map.getTable());

        assertEquals(oneTwoThreeFour, map.remove(39));
        assertEquals(0, map.size());
        expected[0] = null;
        assertArrayEquals(expected, map.getTable());

        assertEquals(zeroOne, stringMap.remove("BB"));
        assertEquals(1, stringMap.size());
        ExternalChainingMapEntry<String, String> exp[] =
                (ExternalChainingMapEntry<String, String>[]) new ExternalChainingMapEntry[13];
        exp[6] = new ExternalChainingMapEntry<>("Aa", zero);
        assertArrayEquals(exp, stringMap.getTable());

        assertEquals(zero, stringMap.remove("Aa"));
        assertEquals(0, stringMap.size());
        exp[6] = null;
        assertArrayEquals(exp, stringMap.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testExternalChainingContainsKey() {
        assertNull(map.put(0, "1"));
        assertNull(map.put(13, "1.2"));
        assertNull(map.put(26, "1.2.3"));
        assertEquals(3, map.size());

        assertTrue(map.containsKey(0));
        assertTrue(map.containsKey(13));
        assertTrue(map.containsKey(26));
        assertFalse(map.containsKey(1));
    }
    @Test(timeout = TIMEOUT)
    public void testExternalChainingGetKeys() {
        assertNull(map.put(0, "1"));
        assertNull(map.put(13, "1.2"));
        assertNull(map.put(26, "1.2.3"));
        assertNull(map.put(1, "2"));
        assertEquals(4, map.size());

        Set<Integer> expected = new HashSet<>();
        expected.add(26);
        expected.add(13);
        expected.add(0);
        expected.add(1);

        assertEquals(expected, map.keySet());
    }

    @Test(timeout = TIMEOUT)
    public void testExternalChainingGetValues() {
        assertNull(map.put(0, "1"));
        assertNull(map.put(13, "1.2"));
        assertNull(map.put(26, "1.2.3"));
        assertNull(map.put(1, "2"));
        assertEquals(4, map.size());

        List<String> expected = new ArrayList<>();
        expected.add("1.2.3");
        expected.add("1.2");
        expected.add("1");
        expected.add("2");

        assertEquals(expected, map.values());
    }

    @Test(timeout = TIMEOUT)
    public void testExternalChainingResize() {
        assertNull(map.put(0, "1"));
        assertNull(map.put(13, "1.2"));
        assertNull(map.put(26, "1.2.3"));
        for (int i = 1; i < 6; i++) {
            assertNull(map.put(i, Integer.toString(i + 1)));
        }
        assertEquals(8, map.size());

        assertNull(map.put(6, "7"));
        ExternalChainingMapEntry<Integer, String> expected[] =
                (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[27];
        for (int i = 0; i < 7; i++) {
            expected[i] = new ExternalChainingMapEntry<>(i, Integer.toString(i + 1));
        }
        expected[13] = new ExternalChainingMapEntry<>(13, "1.2");
        expected[26] = new ExternalChainingMapEntry<>(26, "1.2.3");

        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    public void testNegativeKey() {
        String temp = "1";
        assertNull(map.put(-1, temp));

        assertEquals(1, map.size());

        ExternalChainingMapEntry<Integer, String> expected[] =
                (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];

        expected[1] = new ExternalChainingMapEntry<>(-1, "1");
        assertArrayEquals(expected, map.getTable());

        assertEquals("1", map.get(-1));
        assertEquals(temp, map.remove(-1));
        assertEquals(0, map.size());
    }

    @Test(timeout = TIMEOUT)
    public void testPutExceptions() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> map.put(null, "1"));
        assertNotNull(exception.getMessage());

        exception = assertThrows(IllegalArgumentException.class, () -> map.put(0, null));
        assertNotNull(exception.getMessage());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveExceptions() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> map.remove(null));
        assertNotNull(exception.getMessage());

        exception = assertThrows(NoSuchElementException.class, () -> map.remove(1));
        assertNotNull(exception.getMessage());

        assertNull(map.put(0, "1"));
        assertNull(map.put(13, "2"));
        assertNull(map.put(26, "3"));

        exception = assertThrows(NoSuchElementException.class, () -> map.remove(39));
        assertNotNull(exception.getMessage());
    }

    @Test(timeout = TIMEOUT)
    public void testGetExceptions() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> map.get(null));
        assertNotNull(exception.getMessage());

        exception = assertThrows(NoSuchElementException.class, () -> map.get(1));
        assertNotNull(exception.getMessage());

        assertNull(map.put(0, "1"));
        assertNull(map.put(13, "2"));
        assertNull(map.put(26, "3"));

        exception = assertThrows(NoSuchElementException.class, () -> map.get(39));
        assertNotNull(exception.getMessage());
    }

    @Test(timeout = TIMEOUT)
    public void testContainsKeyExceptions() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> map.containsKey(null));
        assertNotNull(exception.getMessage());
    }

    @Test(timeout = TIMEOUT)
    public void testResizeBackingTableExceptions() {
        map.put(0, "1");
        map.put(1, "2");

        Exception exception = assertThrows(IllegalArgumentException.class, () -> map.resizeBackingTable(1));
        assertNotNull(exception.getMessage());

        exception = assertThrows(IllegalArgumentException.class, () -> map.resizeBackingTable(-1));
        assertNotNull(exception.getMessage());
    }
}