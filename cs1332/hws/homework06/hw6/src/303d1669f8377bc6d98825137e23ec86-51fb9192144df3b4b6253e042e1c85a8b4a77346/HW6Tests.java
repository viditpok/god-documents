import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertThrows;

import org.junit.Before;
import org.junit.Test;

import java.util.*;

public class HW6Tests {
    private static final int TIMEOUT = 200;
    private ExternalChainingHashMap<Integer, String> map;

    @Before
    public void setUp() {
        map = new ExternalChainingHashMap<>();
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutExceptions() {

        assertThrows(IllegalArgumentException.class, () -> {
            map.put(null, "A"); //null key
        });

        assertThrows(IllegalArgumentException.class, () -> {
            map.put(5, null); //null value
        });
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutRegular() {

        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));

        assertEquals(3, map.size());
        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutNegativeKey() {

        assertNull(map.put(0, "A"));
        assertNull(map.put(-1, "B"));
        assertNull(map.put(-2, "C"));

        assertEquals(3, map.size());
        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(-1, "B");
        expected[2] = new ExternalChainingMapEntry<>(-2, "C");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutResize() {

        //length = 13
        //resize happens when 9th gonna be added

        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(14, "G")); //Should link to index 1
        assertNull(map.put(6, "H"));
        assertNull(map.put(7, "I")); //Should cause resize

        assertEquals(9, map.size());
        assertEquals(27, map.getTable().length);

        //rehash, so index 14 should have G now:

        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[27];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");
        expected[5] = new ExternalChainingMapEntry<>(5, "F");
        expected[6] = new ExternalChainingMapEntry<>(6, "H");
        expected[7] = new ExternalChainingMapEntry<>(7, "I");
        expected[14] = new ExternalChainingMapEntry<>(14, "G");

        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutResizeWithDuplicateAndThenCollision() {

        //tests all three
        //length = 13
        //

        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D")); //changed to E because of next line
        assertEquals("D", map.put(3, "E")); //Duplicate key with 3, changes 3's value to E
        assertNull(map.put(13, "F")); //Should collide to index 0
        assertNull(map.put(14, "G")); //Should collide to index 1
        assertNull(map.put(15, "H")); //Should collide to index 2

        //would be A | F , B | G, C | H, E, null, null, null before resize

        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(13, "F", new ExternalChainingMapEntry<>(0, "A"));
        expected[1] = new ExternalChainingMapEntry<>(14, "G", new ExternalChainingMapEntry<>(1, "B"));
        expected[2] = new ExternalChainingMapEntry<>(15, "H", new ExternalChainingMapEntry<>(2, "C"));
        expected[3] = new ExternalChainingMapEntry<>(3, "E");
        //collided ones added to head

        assertEquals(7, map.size());
        assertEquals(13, map.getTable().length);
        assertArrayEquals(expected, map.getTable());

        assertNull(map.put(7, "I")); //Should still cause resize, even though size is not 9
        assertNull(map.put(8, "J")); //Should still cause resize, even though size is not 9

        //rehash now

        ExternalChainingMapEntry<Integer, String>[] expected2 = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[27];
        expected2[0] = new ExternalChainingMapEntry<>(0, "A");
        expected2[1] = new ExternalChainingMapEntry<>(1, "B", new ExternalChainingMapEntry<>(14, "G"));
        expected2[2] = new ExternalChainingMapEntry<>(2, "C", new ExternalChainingMapEntry<>(15, "H"));
        expected2[3] = new ExternalChainingMapEntry<>(3, "E");
        expected2[7] = new ExternalChainingMapEntry<>(7, "I");
        expected2[8] = new ExternalChainingMapEntry<>(8, "J");
        expected2[13] = new ExternalChainingMapEntry<>(13, "F");
        expected2[14] = new ExternalChainingMapEntry<>(14, "G");
        expected2[15] = new ExternalChainingMapEntry<>(15, "H");

        assertEquals(9, map.size());
        assertEquals(27, map.getTable().length);

        assertArrayEquals(expected2, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveExceptions() {

        assertThrows(IllegalArgumentException.class, () -> {
            map.remove(null); //null key
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.put(0, "A");
            map.put(1, "B");
            map.put(2, "C");
            map.put(3, "D");
            map.remove(4); //no 4 key hash
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.put(0, "A");
            map.put(1, "B");
            map.put(2, "C");
            map.put(3, "D");
            map.remove(14); //there is a key hash for 14, but no key itself
        });
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveRegular() {

        map.put(0, "A");
        map.put(1, "B");
        map.put(2, "C");
        map.put(3, "D");

        assertEquals("D", map.remove(3));
        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveFromHeadOfLL() {

        map.put(0, "A");
        map.put(1, "B");
        map.put(2, "C");
        map.put(3, "D");
        map.put(14, "E"); //chains to index 1
        map.put(27, "F"); //chains to index 1 too

        assertEquals("F", map.remove(27)); //removes head
        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(14, "E", new ExternalChainingMapEntry<>(1, "B"));
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveFromRestOfLL() {

        map.put(0, "A");
        map.put(1, "B");
        map.put(2, "C");
        map.put(3, "D");
        map.put(14, "E"); //chains to index 1
        map.put(27, "F"); //chains to index 1 too
        map.put(40, "G"); //chains to index 1 too

        assertEquals("E", map.remove(14));
        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(40, "G", new ExternalChainingMapEntry<>(27, "F", new ExternalChainingMapEntry<>(1, "B")));
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        assertArrayEquals(expected, map.getTable());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveAfterResize() {

        map.put(0, "A");
        map.put(1, "B");
        map.put(2, "C");
        map.put(3, "D");
        map.put(14, "E"); //chains to index 1
        map.put(27, "F"); //chains to index 1 too
        map.put(40, "G"); //chains to index 1 too
        map.put(15, "H"); //chains to index 2
        map.put(4, "I"); //causes resize

        //rehash
        //14 goes to 14
        //40 goes to 13
        //15 goes to 15
        //27 goes to 0

        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[27];
        expected[0] = new ExternalChainingMapEntry<>(27, "F", new ExternalChainingMapEntry<>(0, "A"));
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "I");
        expected[13] = new ExternalChainingMapEntry<>(40, "G");
        expected[14] = new ExternalChainingMapEntry<>(14, "E");
        expected[15] = new ExternalChainingMapEntry<>(15, "H");

        assertArrayEquals(expected, map.getTable());

        assertEquals("A", map.remove(0));
        assertEquals("I", map.remove(4));

        ExternalChainingMapEntry<Integer, String>[] expected2 = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[27];
        expected2[0] = new ExternalChainingMapEntry<>(27, "F");
        expected2[1] = new ExternalChainingMapEntry<>(1, "B");
        expected2[2] = new ExternalChainingMapEntry<>(2, "C");
        expected2[3] = new ExternalChainingMapEntry<>(3, "D");
        expected2[13] = new ExternalChainingMapEntry<>(40, "G");
        expected2[14] = new ExternalChainingMapEntry<>(14, "E");
        expected2[15] = new ExternalChainingMapEntry<>(15, "H");

        assertArrayEquals(expected2, map.getTable());

    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveAll() {

        map.put(0, "A");
        map.put(1, "B");
        map.put(2, "C");
        map.put(3, "D");

        map.remove(0);
        map.remove(1);
        map.remove(2);
        map.remove(3);

        assertThrows(NoSuchElementException.class, () -> {
            map.remove(3); //can't remove 3 again
        });

        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];

        assertArrayEquals(expected, map.getTable()); //empty array

    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void getExceptions() {

        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");

        assertThrows(IllegalArgumentException.class, () -> {
            map.get(null); //null key
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.get(4); //hash isn't there
        });

        assertThrows(NoSuchElementException.class, () -> {
            map.get(14); //hash is there but key isn't
        });

    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void getOneElementLL() {

        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");

        assertEquals("C", map.get(3));

    }

    @SuppressWarnings("unchecked")
    public void getFromHeadLL() {

        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(14, "D"); //chain to 1
        map.put(27, "E"); //chain to 1
        map.put(40, "F"); //chain to 1

        assertEquals("F", map.get(40)); //head is 40
    }

    @SuppressWarnings("unchecked")
    public void getFromRestOfLL() {

        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(14, "D"); //chain to 1
        map.put(27, "E"); //chain to 1
        map.put(40, "F"); //chain to 1

        assertEquals("E", map.get(27)); //middle is 27
    }

    @SuppressWarnings("unchecked")
    public void getFromEndOfLL() {

        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(14, "D"); //chain to 1
        map.put(27, "E"); //chain to 1
        map.put(40, "F"); //chain to 1

        assertEquals("A", map.get(1)); //end is 1
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void containsFunctionality() {
        map.put(1, "A");
        map.put(2, "B");
        map.put(3, "C");
        map.put(4, "D");
        map.put(27, "E"); //chained to 1

        assertFalse(map.containsKey(14)); //hash is there but key isn't
        assertFalse(map.containsKey(5)); //key isn't there

        assertTrue(map.containsKey(2)); //key is there
        assertTrue(map.containsKey(27)); //key is there, chained to the 1st index
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void keySetFunctionalityWithMultipleOperations() {

        map.put(0, "A");
        map.put(1, "B");
        map.put(2, "C");
        map.put(3, "D");
        map.put(14, "E"); //chains to index 1
        map.remove(1); //removes 1
        map.put(27, "F"); //chains to index 1 too
        map.put(40, "G"); //chains to index 1 too
        map.put(15, "H"); //chains to index 2
        map.remove(0); //removes 0
        map.put(4, "I"); //causes resize

        Set<Integer> expectedSet = new HashSet<>();

        expectedSet.add(2);
        expectedSet.add(3);
        expectedSet.add(14);
        expectedSet.add(27);
        expectedSet.add(40);
        expectedSet.add(15);
        expectedSet.add(4);

        //order doesn't matter

        assertEquals(expectedSet, map.keySet());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void valuesFunctionalityWithMultipleOperations() {

        map.put(0, "A");
        map.put(1, "B");
        map.put(2, "C");
        map.put(3, "D");
        map.put(14, "E"); //chains to index 1
        map.remove(1); //removes 1
        map.put(27, "F"); //chains to index 1 too
        map.put(40, "G"); //chains to index 1 too
        map.put(15, "H"); //chains to index 2
        map.remove(0); //removes 0
        map.put(4, "I"); //causes resize

        //rehash
        //14 goes to 14
        //40 goes to 13
        //15 goes to 15
        //27 goes to 0

        LinkedList<String> expectedValues = new LinkedList<>();

        expectedValues.add("G");
        expectedValues.add("F");
        expectedValues.add("E");
        expectedValues.add("H");
        expectedValues.add("C");
        expectedValues.add("D");
        expectedValues.add("I");

        //in this order because that's the order it iterates in

        assertEquals(expectedValues, map.values());
    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void resizeToSmallerLengthExceptionButEqualIsAllowed() {
        map.put(0, "A");
        map.put(1, "B");
        map.put(2, "C");
        map.put(3, "D");
        //size is 4, so 4 is allowed but 3 isn't

        map.resizeBackingTable(4);

        assertThrows(IllegalArgumentException.class, () -> {
            map.resizeBackingTable(3);
        });

    }

    public void resizeFunctionality() {
        map.put(0, "A");
        map.put(1, "B");
        map.put(2, "C");
        map.put(3, "D");
        map.put(4, "E");
        //size is 4, so 4 is allowed but 3 isn't

        map.resizeBackingTable(6);

        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[6];

        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");

        //length of expected is 6

        assertArrayEquals(expected, map.getTable());


    }

    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void clear() {

        map.put(0, "A");
        map.put(1, "B");
        map.put(2, "C");
        map.put(3, "D");

        map.clear();

        ExternalChainingMapEntry<Integer, String>[] expected = (ExternalChainingMapEntry<Integer, String>[]) new ExternalChainingMapEntry[13];

        assertArrayEquals(expected, map.getTable());
    }

}