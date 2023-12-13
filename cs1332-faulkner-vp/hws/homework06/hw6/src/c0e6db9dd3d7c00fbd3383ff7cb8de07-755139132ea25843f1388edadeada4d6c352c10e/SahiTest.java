import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;

public class SahiTest {
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
    public void testPutNoNullNoResize () {
        // [(0, A), (1, B), (2, C), _, _]
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertTrue(map.containsKey(1));
        assertFalse(map.containsKey(336));
        assertEquals("C", map.get(2));
        
        Set<Integer> expectedKeySet = new HashSet<>();
        expectedKeySet.add(0);
        expectedKeySet.add(1);
        expectedKeySet.add(2);
        assertEquals(expectedKeySet, map.keySet());

        assertEquals(3, map.size());
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[5];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutNegativeKeyNoNullNoResize () {
        // [(0, A), (-1, B), (-2, C), _, _]
        assertNull(map.put(0, "A"));
        assertNull(map.put(-1, "B"));
        assertEquals("B", map.get(-1));
        assertNull(map.put(-2, "C"));
        
        Set<Integer> expectedKeySet = new HashSet<>();
        expectedKeySet.add(0);
        expectedKeySet.add(-1);
        expectedKeySet.add(-2);
        assertEquals(expectedKeySet, map.keySet());

        assertEquals(3, map.size());
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[5];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(-1, "B");
        expected[2] = new ExternalChainingMapEntry<>(-2, "C");
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutNullFirstIndexNoResize () {
        // [_, (1, B), (2, C), _, _]
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertTrue(map.containsKey(2));
        assertFalse(map.containsKey(11));
        assertEquals("B", map.get(1));

        assertEquals(2, map.size());
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[5];
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutConsecutiveNullNoResize () {
        // [_, _, (2, C), _, _]
        
        
        assertNull(map.put(2, "C"));

        assertEquals(1, map.size());
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[5];
      
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutNoNullResize() {
        // [(0, A), (1, B), (2, C), (3, D), (4, E), _, _, _, _, _, _]
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertEquals("A", map.get(0));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));

        assertEquals(5, map.size());
        assertEquals(11, map.getTable().length);
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[11];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");

        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutOneNullResize() {
        // [(0, A), (1, B), (2, C), _, (4, E), _, _, _, _, _, _, _, _]
        
    	assertNull(map.put(0, "A"));
    	assertNull(map.put(1, "B"));
    	assertNull(map.put(2, "C"));
        assertNull(map.put(4, "E"));
        assertEquals("E", map.get(4));

        assertEquals(4, map.size());
        assertEquals(11, map.getTable().length);
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[11];
        
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");

        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutConsecutiveNullResize() {
    	 // [_, (1, B), (2, C), (3, D), (4, E), _, _, _, _, _, (10, K)]
        
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(10, "K"));
        
        Set<Integer> expectedKeySet = new HashSet<>();
        expectedKeySet.add(1);
        expectedKeySet.add(2);
        expectedKeySet.add(3);
        expectedKeySet.add(4);
        expectedKeySet.add(10);
        assertEquals(expectedKeySet, map.keySet());
        
        assertEquals(5, map.size());
        assertEquals(11, map.getTable().length);
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[11];
        
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");
        expected[10] = new ExternalChainingMapEntry<>(10, "K");

        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutNoNullLinked() {
    	// [(11, A_1), (12, B_1), (13, C_1), _, _, _, _, _, _, _, _]
    	
    	// (11, A_1) --> (0, A)
    	// (12, B_1) --> (1, B)
    	// (13, C_1) --> (2, C)
    	
    	assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        
        assertNull(map.put(11, "A_1"));
        assertNull(map.put(12, "B_1"));
        assertNull(map.put(13,"C_1"));
        assertEquals("B_1", map.get(12));
        assertEquals("C", map.get(2));
        
        Set<Integer> expectedKeySet = new HashSet<>();
        expectedKeySet.add(11);
        expectedKeySet.add(0);
        expectedKeySet.add(12);
        expectedKeySet.add(1);
        expectedKeySet.add(13);
        expectedKeySet.add(2);
        assertEquals(expectedKeySet, map.keySet());
        
        assertEquals(6, map.size());
        
        assertEquals(11, map.getTable().length);
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[11];
        
        expected[0] = new ExternalChainingMapEntry<>(11, "A_1");
        assertEquals(new ExternalChainingMapEntry<>(0, "A"), map.getTable()[0].getNext());
        expected[1] = new ExternalChainingMapEntry<>(12, "B_1");
        assertEquals(new ExternalChainingMapEntry<>(1, "B"), map.getTable()[1].getNext());
        expected[2] = new ExternalChainingMapEntry<>(13, "C_1");
        assertEquals(new ExternalChainingMapEntry<>(2, "C"), map.getTable()[2].getNext());
        
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutBeginningNullLinked() {
    	// [_, (12, B_1), (13, C_1), _, _, _, _, _, _, _, _]
    	
    	
    	// (12, B_1) --> (1, B)
    	// (13, C_1) --> (2, C)
    	
    	
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        
        
        assertNull(map.put(12, "B_1"));
        assertNull(map.put(13,"C_1"));
        
        Set<Integer> expectedKeySet = new HashSet<>();
        expectedKeySet.add(12);
        expectedKeySet.add(1);
        expectedKeySet.add(13);
        expectedKeySet.add(2);
        assertEquals(expectedKeySet, map.keySet());
        
        assertEquals(4, map.size());
        
        assertEquals(11, map.getTable().length);
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[11];
        
      
        expected[1] = new ExternalChainingMapEntry<>(12, "B_1");
        assertEquals(new ExternalChainingMapEntry<>(1, "B"), map.getTable()[1].getNext());
        expected[2] = new ExternalChainingMapEntry<>(13, "C_1");
        assertEquals(new ExternalChainingMapEntry<>(2, "C"), map.getTable()[2].getNext());
        
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutOneNullLinked() {
    	// [(11, A_1), _, (13, C_1), _, _, _, _, _, _, _, _]
    	
    	// (11, A_1) --> (0, A)
    	
    	// (13, C_1) --> (2, C)
    	
    	assertNull(map.put(0, "A"));
        
        assertNull(map.put(2, "C"));
        
        assertNull(map.put(11, "A_1"));
        
        assertNull(map.put(13,"C_1"));
        
        assertTrue(map.containsKey(13));
        assertFalse(map.containsKey(-4));
        
        assertEquals(4, map.size());
        
        assertEquals(11, map.getTable().length);
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[11];
        
        expected[0] = new ExternalChainingMapEntry<>(11, "A_1");
        assertEquals(new ExternalChainingMapEntry<>(0, "A"), map.getTable()[0].getNext());
        expected[2] = new ExternalChainingMapEntry<>(13, "C_1");
        assertEquals(new ExternalChainingMapEntry<>(2, "C"), map.getTable()[2].getNext());
        
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutGapNullLinked() {
    	// [(11, A_1), (12, B_1), _, _, (15, E), _, _, _, _, _, _]
    	
    	// (11, A_1) --> (0, A)
    	// (12, B_1) --> (1, B)
    	// (15, E_1) --> (4, E)
    	
    	assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(4, "E"));
        
        assertNull(map.put(11, "A_1"));
        assertNull(map.put(12, "B_1"));
        assertNull(map.put(15, "E_1"));
        assertEquals("A", map.get(0));
        assertEquals("E_1", map.get(15));
        assertEquals("B", map.get(1));
        
        assertEquals(6, map.size());
        
        assertEquals(11, map.getTable().length);
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[11];
        
        expected[0] = new ExternalChainingMapEntry<>(11, "A_1");
        assertEquals(new ExternalChainingMapEntry<>(0, "A"), map.getTable()[0].getNext());
        expected[1] = new ExternalChainingMapEntry<>(12, "B_1");
        assertEquals(new ExternalChainingMapEntry<>(1, "B"), map.getTable()[1].getNext());
        expected[4] = new ExternalChainingMapEntry<>(15, "E_1");
        assertEquals(new ExternalChainingMapEntry<>(4, "E"), map.getTable()[4].getNext());
        
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutOneLinkedNoResize() {
    	// [_, _, (12, C_2), _, _]
    	
    	// (12, C_2) --> (7, C_1) --> (2, C)
    	
        assertNull(map.put(2, "C"));
        assertNull(map.put(7, "C_1"));
        assertNull(map.put(12, "C_2"));
        assertEquals("C_1", map.get(7));
        
        Set<Integer> expectedKeySet = new HashSet<>();
        expectedKeySet.add(12);
        expectedKeySet.add(7);
        expectedKeySet.add(2);
        assertEquals(expectedKeySet, map.keySet());
        
        assertEquals(3, map.size());
        
        assertEquals(5, map.getTable().length);
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[5];
        
        expected[2] = new ExternalChainingMapEntry<>(12, "C_2");
        assertEquals(new ExternalChainingMapEntry<>(7, "C_1"), map.getTable()[2].getNext());
        assertEquals(new ExternalChainingMapEntry<>(2, "C"), map.getTable()[2].getNext().getNext());
        
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutNoResizeDuplicates() {
        // [(0, A_1), _, (2, C), _, _]
        assertNull(map.put(0, "A"));
        assertEquals("A", map.put(0, "A_1"));
        assertNull(map.put(2, "C"));

        assertEquals(2, map.size());
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[5];
        expected[0] = new ExternalChainingMapEntry<>(0, "A_1");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutResizeDuplicates() {
        // [(0, A), (1, B_1), (2, C), (3, D_1), _, _, _, _, _, _, _]
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertEquals("D", map.put(3, "D_1"));
        assertEquals("B", map.put(1, "B_1"));

        assertEquals(4, map.size());
        assertEquals(11, map.getTable().length);
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[11];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B_1");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(3, "D_1");
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testPutGapNullLinkedDuplicate() {
    	// [(11, A_1), _, _, _, (15, E), _, _, _, _, _, _]
    	
    	// (11, A_1) --> (0, A)
    	// (15, E_1) --> (4, E!)
    	
    	assertNull(map.put(0, "A"));
        assertNull(map.put(4, "E"));
        
        assertNull(map.put(11, "A_1"));
        assertNull(map.put(15, "E_1"));
        assertEquals("E", map.put(4, "E!"));
        
        assertEquals(4, map.size());
        
        assertEquals(11, map.getTable().length);
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[11];
        
        expected[0] = new ExternalChainingMapEntry<>(11, "A_1");
        assertEquals(new ExternalChainingMapEntry<>(0, "A"), map.getTable()[0].getNext());
        expected[4] = new ExternalChainingMapEntry<>(15, "E_1");
        assertEquals(new ExternalChainingMapEntry<>(4, "E!"), map.getTable()[4].getNext());
        
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testPutNullKey() {
    	assertNull(map.put(null, "A"));
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testPutNullValue() {
    	assertNull(map.put(12, null));
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveOneNoLinkedNoResize() {
        String temp = "A";

        // [(0, A), (1, B), (2, C), _, _]
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));

        assertEquals(3, map.size());

        // [_, (1, B), (2, C), _, _]
        assertSame(temp, map.remove(0));

        assertEquals(2, map.size());

        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[5];
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveMultipleNoLinkedNoResize() {
        String temp = "A";
        String temp2 = "B";

        // [(0, A), (1, B), (2, C), _, _]
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));

        assertEquals(3, map.size());

        // [_, _, (2, C), _, _]
        assertSame(temp, map.remove(0));
        assertSame(temp2, map.remove(1));
        
        assertEquals(1, map.size());

        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[5];
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveMultipleNoLinkedResize() {
        // [(0, A), (1, B), (2, C), (3, D), (4, E), (5, F), (6, G), _, _, _, _]
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(4, "E"));
        assertNull(map.put(5, "F"));
        assertNull(map.put(6, "G"));
        
        List<String> expectedValues = new LinkedList<>();
        expectedValues.add("A");
        expectedValues.add("B");
        expectedValues.add("C");
        expectedValues.add("D");
        expectedValues.add("E");
        expectedValues.add("F");
        expectedValues.add("G");
        assertEquals(expectedValues, map.values());

        assertTrue(map.containsKey(2)); 
        
        // [(0, A), (1, B), _, (3, D), (4, E), _, _, _, _, _, _]
        assertSame("C", map.remove(2));
        assertSame("F", map.remove(5));
        assertSame("G", map.remove(6));
        
        assertEquals(4, map.size());
        assertEquals(11, map.getTable().length);
        
        ExternalChainingMapEntry<Integer, String>[] expected =
            (ExternalChainingMapEntry<Integer, String>[])
                new ExternalChainingMapEntry[11];
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = null;
        expected[3] = new ExternalChainingMapEntry<>(3, "D");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");
        expected[5] = null;
        expected[6] = null;

        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveHeadOnlyLinked() {
    	// [(0, A), (1, B), (2, C), (36, D_3), _, _, _, _, _, _, _]
    	
    	// (36, D_3) --> (25, D_2) --> (14, D_1) --> (3, D)
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(14, "D_1"));
        assertNull(map.put(25, "D_2"));
        assertNull(map.put(36, "D_3"));
        assertTrue(map.containsKey(25));
        assertTrue(map.containsKey(14));
        
        Set<Integer> expectedKeySet = new HashSet<>();
        expectedKeySet.add(0);
        expectedKeySet.add(1);
        expectedKeySet.add(2);
        expectedKeySet.add(36);
        expectedKeySet.add(25);
        expectedKeySet.add(14);
        expectedKeySet.add(3);
        assertEquals(expectedKeySet, map.keySet());
        
        assertEquals(7, map.size());
        assertEquals(11, map.getTable().length);
        
        // [(0, A), (1, B), (2, C), (25, D_2), _, _, _, _, _, _, _]
    	
    	// (25, D_2) --> (14, D_1) --> (3, D)
        assertSame("D_3", map.remove(36));
        assertEquals(6, map.size());
        
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                    new ExternalChainingMapEntry[11];
        
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(25, "D_2");
        assertEquals(new ExternalChainingMapEntry<>(14, "D_1"), map.getTable()[3].getNext());
        assertEquals(new ExternalChainingMapEntry<>(3, "D"), map.getTable()[3].getNext().getNext());
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveMiddleOnlyLinked() {
    	// [(0, A), (1, B), (2, C), (36, D_3), _, _, _, _, _, _, _]
    	
    	// (36, D_3) --> (25, D_2) --> (14, D_1) --> (3, D)
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(14, "D_1"));
        assertNull(map.put(25, "D_2"));
        assertNull(map.put(36, "D_3"));
        
        assertEquals("D_2", map.get(25));
        assertEquals("D", map.get(3));
        
        assertEquals(7, map.size());
        assertEquals(11, map.getTable().length);
        
        // [(0, A), (1, B), (2, C), (36, D_3), _, _, _, _, _, _, _]
    	
    	// (36, D_3) --> (14, D_1) --> (3, D)
        assertSame("D_2", map.remove(25));
        assertEquals(6, map.size());
        
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                    new ExternalChainingMapEntry[11];
        
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(36, "D_3");
        assertEquals(new ExternalChainingMapEntry<>(14, "D_1"), map.getTable()[3].getNext());
        assertEquals(new ExternalChainingMapEntry<>(3, "D"), map.getTable()[3].getNext().getNext());
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveAllMiddleOnlyLinked() {
    	// [(0, A), (1, B), (2, C), (36, D_3), _, _, _, _, _, _, _]
    	
    	// (36, D_3) --> (25, D_2) --> (14, D_1) --> (3, D)
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(14, "D_1"));
        assertNull(map.put(25, "D_2"));
        assertNull(map.put(36, "D_3"));
        
        assertEquals(7, map.size());
        assertEquals(11, map.getTable().length);
        
        // [(0, A), (1, B), (2, C), (36, D_3), _, _, _, _, _, _, _]
    	
    	// (36, D_3) --> (3, D)
        assertSame("D_2", map.remove(25));
        assertSame("D_1", map.remove(14));
        assertEquals(5, map.size());
        
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                    new ExternalChainingMapEntry[11];
        
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(36, "D_3");
        assertEquals(new ExternalChainingMapEntry<>(3, "D"), map.getTable()[3].getNext());
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveEndOnlyLinked() {
    	// [(0, A), (1, B), (2, C), (36, D_3), _, _, _, _, _, _, _]
    	
    	// (36, D_3) --> (25, D_2) --> (14, D_1) --> (3, D)
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(14, "D_1"));
        assertNull(map.put(25, "D_2"));
        assertNull(map.put(36, "D_3"));
        
        assertEquals(7, map.size());
        assertEquals(11, map.getTable().length);
        
        // [(0, A), (1, B), (2, C), (36, D_3), _, _, _, _, _, _, _]
    	
        // (36, D_3) --> (25, D_2) --> (14, D_1)
        assertSame("D", map.remove(3));
        assertEquals(6, map.size());
        
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                    new ExternalChainingMapEntry[11];
        
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = new ExternalChainingMapEntry<>(36, "D_3");
        assertEquals(new ExternalChainingMapEntry<>(25, "D_2"), map.getTable()[3].getNext());
        assertEquals(new ExternalChainingMapEntry<>(14, "D_1"), map.getTable()[3].getNext().getNext());
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveAllOnlyLinked() {
    	// [(0, A), (1, B), (2, C), (36, D_3), _, _, _, _, _, _, _]
    	
    	// (36, D_3) --> (25, D_2) --> (14, D_1) --> (3, D)
        assertNull(map.put(0, "A"));
        assertNull(map.put(1, "B"));
        assertNull(map.put(2, "C"));
        assertNull(map.put(3, "D"));
        assertNull(map.put(14, "D_1"));
        assertNull(map.put(25, "D_2"));
        assertNull(map.put(36, "D_3"));
        
        List<String> expectedValues1 = new LinkedList<>();
        expectedValues1.add("A");
        expectedValues1.add("B");
        expectedValues1.add("C");
        expectedValues1.add("D_3");
        expectedValues1.add("D_2");
        expectedValues1.add("D_1");
        expectedValues1.add("D");
        assertEquals(expectedValues1, map.values());
        
        assertEquals(7, map.size());
        assertEquals(11, map.getTable().length);
        
        // [(0, A), (1, B), (2, C), _, _, _, _, _, _, _, _]
    	
        assertSame("D_1", map.remove(14));
        assertSame("D_3", map.remove(36));
        assertSame("D_2", map.remove(25));
        assertSame("D", map.remove(3));
        assertEquals(3, map.size());
        
        List<String> expectedValues2 = new LinkedList<>();
        expectedValues2.add("A");
        expectedValues2.add("B");
        expectedValues2.add("C");
        assertEquals(expectedValues2, map.values());
        
        Set<Integer> expectedKeySet = new HashSet<>();
        expectedKeySet.add(0);
        expectedKeySet.add(1);
        expectedKeySet.add(2);
        assertEquals(expectedKeySet, map.keySet());
        
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                    new ExternalChainingMapEntry[11];
        
        expected[0] = new ExternalChainingMapEntry<>(0, "A");
        expected[1] = new ExternalChainingMapEntry<>(1, "B");
        expected[2] = new ExternalChainingMapEntry<>(2, "C");
        expected[3] = null;
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveOneElement() {
    	assertNull(map.put(0, "A"));
    	assertEquals(1, map.size());
    	
    	assertSame("A", map.remove(0));
        assertEquals(0, map.size());
         
        ExternalChainingMapEntry<Integer, String>[] expected =
                 (ExternalChainingMapEntry<Integer, String>[])
                     new ExternalChainingMapEntry[5];
         
        expected[0] = null;
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testRemoveWithGaps() {
    	// [(11, A_1), _, _, (36, D_3), (4, E), _, (17, G_1), _, _, _, _]
    	
    	// (11, A_1) --> (0, A)
    	// (36, D_3) --> (14, D_1) 
    	// (17, G_1) --> (6, G)
    	
        assertNull(map.put(0, "A"));
        assertNull(map.put(4, "E")); 
        assertNull(map.put(14, "D_1"));
        assertNull(map.put(36, "D_3"));
        assertNull(map.put(6, "G"));
        assertNull(map.put(17, "G_1"));
        assertNull(map.put(11, "A_1"));
        
        assertEquals(7, map.size());
        assertEquals(11, map.getTable().length);
        
        // [(11, A_1), _, _, (14, D_1), (4, E), _, (6, G), _, _, _, _]
    	
        assertSame("D_3", map.remove(36));
        assertSame("A", map.remove(0));
        assertSame("G_1", map.remove(17));
        assertEquals(4, map.size());
        
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                    new ExternalChainingMapEntry[11];
        
        expected[0] = new ExternalChainingMapEntry<>(11, "A_1");
        expected[3] = new ExternalChainingMapEntry<>(14, "D_1");
        expected[4] = new ExternalChainingMapEntry<>(4, "E");
        expected[6] = new ExternalChainingMapEntry<>(6, "G");
        assertArrayEquals(expected, map.getTable());
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testChainedResize() {
    	// [_, _, _, (14, D_2), _, _, (6, G), _, (30, !), _, _]
    	
    	// (14, D_2) --> (36, D_1) --> (3, D)
    	// (30, !) --> (19, %) --> (8, @)
    	assertNull(map.put(3, "D"));
    	assertNull(map.put(8, "#"));
    	assertNull(map.put(6, "G"));
    	assertNull(map.put(36, "D_1"));
    	assertNull(map.put(14, "D_2"));
    	assertEquals("#", map.put(8, "@"));
    	assertNull(map.put(19, "%"));
    	assertNull(map.put(30, "!"));
    	
    	assertEquals(7, map.size());
        assertEquals(11, map.getTable().length);
        
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                    new ExternalChainingMapEntry[11];

        expected[3] = new ExternalChainingMapEntry<>(14, "D_2");
        assertEquals(new ExternalChainingMapEntry<>(36, "D_1"), map.getTable()[3].getNext());
        assertEquals(new ExternalChainingMapEntry<>(3, "D"), map.getTable()[3].getNext().getNext());
        expected[6] = new ExternalChainingMapEntry<>(6, "G");
        expected[8] = new ExternalChainingMapEntry<>(30, "!");
        assertEquals(new ExternalChainingMapEntry<>(19, "%"), map.getTable()[8].getNext());
        assertEquals(new ExternalChainingMapEntry<>(8, "@"), map.getTable()[8].getNext().getNext());
        assertArrayEquals(expected, map.getTable());
    
    }
    
    @Test(timeout = TIMEOUT)
    @SuppressWarnings("unchecked")
    public void testMultipleOperations() {
    	// [_, _, _, (14, D_2), _, _, (6, G), _, (30, !), _, _]
    	
    	// (14, D_2) --> (36, D_1) --> (3, D)
    	// (30, !) --> (19, %) --> (8, @)
    	assertNull(map.put(3, "D"));
    	assertNull(map.put(8, "#"));
    	assertNull(map.put(6, "G"));
    	assertNull(map.put(36, "D_1"));
    	assertNull(map.put(14, "D_2"));
    	assertEquals("#", map.put(8, "@"));
    	assertNull(map.put(19, "%"));
    	assertNull(map.put(30, "!"));
    	
    	assertEquals(7, map.size());
        assertEquals(11, map.getTable().length);
        
        // [_, _, _, (26, U), _, (75, G_2), (7, hello), (54, $), ...., (59, J), ...., (19, %), ...]
    	
    	// (26, U) --> (3, D)
        // (75, G_2) --> (29, G_1) --> (52, ^)
        // (54, $) --> (31, &)
    	// (59, J) --> (36, DD) 
        // (19, %)
        
        assertEquals("!", map.remove(30));
        assertNull(map.put(31, "&"));
        assertEquals("D_1", map.put(36, "DD"));
        assertEquals("D_2", map.remove(14));
        assertEquals("G", map.remove(6));
        assertNull(map.put(7, "hello"));
        assertEquals("@", map.remove(8));
        assertNull(map.put(52, "^"));
        assertNull(map.put(29, "G_1"));
        assertNull(map.put(75, "G_2"));
        assertNull(map.put(54, "$"));
        assertNull(map.put(26, "U"));
        assertNull(map.put(59, "J"));
        
        Set<Integer> expectedKeySet = new HashSet<>();
        expectedKeySet.add(26);
        expectedKeySet.add(3);
        expectedKeySet.add(75);
        expectedKeySet.add(29);
        expectedKeySet.add(52);
        expectedKeySet.add(7);
        expectedKeySet.add(54);
        expectedKeySet.add(31);
        expectedKeySet.add(59);
        expectedKeySet.add(36);
        expectedKeySet.add(19);
        assertEquals(expectedKeySet, map.keySet());
        
        assertSame(11, map.size());
        assertEquals(23, map.getTable().length);
        
        ExternalChainingMapEntry<Integer, String>[] expected =
                (ExternalChainingMapEntry<Integer, String>[])
                    new ExternalChainingMapEntry[23];
        
        expected[3] = new ExternalChainingMapEntry<>(26, "U"); 
        assertEquals(new ExternalChainingMapEntry<>(3, "D"), map.getTable()[3].getNext());
        expected[6] = new ExternalChainingMapEntry<>(75, "G_2");
        assertEquals(new ExternalChainingMapEntry<>(29, "G_1"), map.getTable()[6].getNext());
        assertEquals(new ExternalChainingMapEntry<>(52, "^"), map.getTable()[6].getNext().getNext());
        expected[7] = new ExternalChainingMapEntry<>(7, "hello");
        expected[8] = new ExternalChainingMapEntry<>(54, "$");
        assertEquals(new ExternalChainingMapEntry<>(31, "&"), map.getTable()[8].getNext());
        expected[13] = new ExternalChainingMapEntry<>(59, "J");
        assertEquals(new ExternalChainingMapEntry<>(36, "DD"), map.getTable()[13].getNext());
        expected[19] = new ExternalChainingMapEntry<>(19, "%");
        assertArrayEquals(expected, map.getTable());
        
        
        // [_, _, _, (26, S), _, (75, G_2), _, (54, $), ...., (36, DD), ...., _, ...]
    	
    	// (26, S) --> (3, D)
        // (75, G_2) --> (52, ^)
        // (54, $) 
    	// (36, DD) 
        
        assertEquals("G_1", map.remove(29));
        assertEquals("J", map.remove(59));
        assertEquals("hello", map.remove(7));
        assertEquals("&", map.remove(31));
        assertEquals("%", map.remove(19));
        assertEquals("U", map.put(26, "S"));
        
        assertSame(6, map.size());
        assertEquals(23, map.getTable().length);
        
        expected[3] = new ExternalChainingMapEntry<>(26, "S"); 
        assertEquals(new ExternalChainingMapEntry<>(3, "D"), map.getTable()[3].getNext());
        expected[6] = new ExternalChainingMapEntry<>(75, "G_2");
        assertEquals(new ExternalChainingMapEntry<>(52, "^"), map.getTable()[6].getNext());
        expected[7] = null;
        expected[8] = new ExternalChainingMapEntry<>(54, "$");
        assertNull(map.getTable()[8].getNext());
        expected[13] = new ExternalChainingMapEntry<>(36, "DD");
        expected[19] = null;
        assertArrayEquals(expected, map.getTable());
        
        assertEquals("G_2", map.get(75));
        
        List<String> expectedValues = new LinkedList<>();
        expectedValues.add("S");
        expectedValues.add("D");
        expectedValues.add("G_2");
        expectedValues.add("^");
        expectedValues.add("$");
        expectedValues.add("DD");
        assertEquals(expectedValues, map.values());
        
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testRemoveNullKey() {
    	assertNull(map.put(12, "A"));
    	map.remove(null);
    }
    
    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testRemoveKeyNotFound() {
    	assertNull(map.put(12, "A"));
    	assertNull(map.put(1, "K"));
    	map.remove(3);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testGetNullKey() {
    	assertNull(map.put(12, "A"));
    	map.get(null);
    }
    
    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testGetKeyNotFound() {
    	assertNull(map.put(12, "A"));
    	assertNull(map.put(1, "K"));
    	map.get(15);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testContainsNullKey() {
    	assertNull(map.put(12, "A"));
    	map.containsKey(null);
    }
    
    @Test(timeout = TIMEOUT)
    public void testClear() {
    	map.put(3, "A");
    	map.put(2, "D");
    	map.clear();
    	assertEquals(0, map.size());
    	assertEquals(13, map.getTable().length);
    }
    
}    

