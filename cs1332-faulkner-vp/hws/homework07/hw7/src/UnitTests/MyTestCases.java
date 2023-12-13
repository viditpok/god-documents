import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.NoSuchElementException;

import static org.junit.Assert.*;

public class MyTestCases {

    private static final int TIMEOUT = 200;
    private AVL<String> tree;
    private List<AVLNode<String>> list;

    // all methods before @Before are used to interpret and judge results of the tests cases

    private String[] getValues() {
        String[] results = new String[list.size()];
        for (int i = 0; i < results.length; ++i) {
            results[i] = list.get(i).getData();
        }
        return results;
    }

    private int[] getHeights() {
        int[] results = new int[list.size()];
        for (int i = 0; i < results.length; ++i) {
            results[i] = list.get(i).getHeight();
        }
        return results;
    }

    private int[] BalanceFactor() {
        int[] results = new int[list.size()];
        for (int i = 0; i < results.length; ++i) {
            results[i] = list.get(i).getBalanceFactor();
        }
        return results;
    }

    private void preorder() {
        list = new ArrayList<>();
        pHelper(tree.getRoot(), list);
    }

    private void pHelper(AVLNode<String> node, List<AVLNode<String>> list) {
        if (node != null) {
            list.add(node);
            pHelper(node.getLeft(), list);
            pHelper(node.getRight(), list);
        }
    }

    @Before
    public void setup() {
        tree = new AVL<>();
    }

    @Test(timeout = TIMEOUT)
    public void testInitialization() {
        //Creates empty tree and checks that size is 0
        assertEquals(0, tree.size());
        assertNull(tree.getRoot());
    }

    @Test(timeout = TIMEOUT)
    public void testConstructorException() {
        //Test creating an AVL with null data
        assertThrows(IllegalArgumentException.class, () -> {
            tree = new AVL<>(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testConstructorException2() {
        //Tests creating an AVL with a null data in the middle
        assertThrows(IllegalArgumentException.class, () -> {
            tree = new AVL<>(Arrays.asList("001", "002", null, "003"));
        });
    }

    @Test(timeout = TIMEOUT)
    public void testHeightOfNullTree() {
        assertEquals(-1, tree.height());
    }

    @Test(timeout = TIMEOUT)
    public void testClear() {
        //Creates a tree and clears it
        tree = new AVL<>(Arrays.asList("9", "8", "7", "6", "5", "4", "3", "2", "1"));
        tree.clear();
        assertNull(tree.getRoot());
        assertEquals(0, tree.size());
    }

    @Test(timeout = TIMEOUT)
    public void testAddException() {
        //Tries to add null to an AVL, exception should be thrown
        assertThrows(IllegalArgumentException.class, () -> {
            tree.add(null);
        });
    }

    //EVERY TEST CASE FOR ADD TESTS ALL 4 POSSIBLE ROTATIONS

    @Test(timeout = TIMEOUT)
    public void testAdd() {
        /**Performs a simple addition that creates the resultsulting tree:
         *
         *          30
         *        /    \
         *      14     44
         *
        */
        tree = new AVL<>(Arrays.asList("44", "14"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(2, tree.size());
        assertArrayEquals(new String[] {"44", "14"}, getValues);
        assertArrayEquals(new int[] {1, 0}, getHeights);
        assertArrayEquals(new int[] {1, 0}, BalanceFactor);
        tree.add("30");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(3, tree.size());
        assertArrayEquals(new String[] {"30", "14", "44"}, getValues);
        assertArrayEquals(new int[] {1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAddWithNegativeNums() {
        /**Performs a simple addition that creates the resultsulting tree:
         *
         *          -14
         *        /     \
         *      -44     -3
         *
         */
        tree = new AVL<>(Arrays.asList("-14", "-44"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(2, tree.size());
        assertArrayEquals(new String[] {"-14", "-44"}, getValues);
        assertArrayEquals(new int[] {1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 0}, BalanceFactor);
        tree.add("-3");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(3, tree.size());
        assertArrayEquals(new String[] {"-3", "-14", "-44"}, getValues);
        assertArrayEquals(new int[] {1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd2() {
        //Creates a tree with 3 (0-2) height levels and tests the add method
        tree = new AVL<>(Arrays.asList("30", "16", "44", "00", "21"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"30", "16", "00", "21", "44"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 0}, getHeights);
        assertArrayEquals(new int[] {1, 0, 0, 0, 0}, BalanceFactor);
        tree.add("27");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"21", "16", "00", "30", "27", "44"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 1, 0, 0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAddWithNegativeNums2() {
        //Creates a tree with 3 (0-2) height levels and tests the add method with negative numbers
        tree = new AVL<>(Arrays.asList("-16", "-30", "00", "-44", "-21"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"-30", "-16", "-21", "00", "-44"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, -1, 0, 1, 0}, BalanceFactor);
        tree.add("-19");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"-30", "-19", "-16", "-21", "00", "-44"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, 1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd3() {
        //Creates a tree with 4 height levels and tests the add method
        tree = new AVL<>(Arrays.asList("23", "07", "29", "06", "12", "24", "33", "05", "10", "14"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(10, tree.size());
        assertArrayEquals(new String[] {"23", "07", "06", "05", "12", "10", "14", "29", "24", "33"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 1, 0, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {1, 0, 1, 0, 0, 0, 0, 0, 0, 0}, BalanceFactor);
        tree.add("18");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"12", "07", "06", "05", "10", "23", "14", "18", "29", "24", "33"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 0, 2, 1, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 1, 1, 0, 0, 0, -1, 0, 0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAddWithNegativeNums3() {
        //Creates a tree with 4 height levels and tests the add method with negative numbers
        tree = new AVL<>(Arrays.asList("-23", "-07", "-29", "-06", "-12", "-24", "-33", "-05", "-10", "-14"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(10, tree.size());
        assertArrayEquals(new String[] {"-23", "-07", "-06", "-05", "-12", "-10", "-14", "-29", "-24", "-33"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 1, 0, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {1, 0, 1, 0, 0, 0, 0, 0, 0, 0}, BalanceFactor);
        tree.add("-18");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"-12", "-07", "-06", "-05", "-10", "-23", "-14", "-18", "-29", "-24", "-33"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 0, 2, 1, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 1, 1, 0, 0, 0, -1, 0, 0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd4() {
        //Creates a tree with 3 height levels and performs different rotation from simpleAdd2
        tree = new AVL<>(Arrays.asList("28", "12", "34", "11", "17"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"28", "12", "11", "17", "34"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 0}, getHeights);
        assertArrayEquals(new int[] {1, 0, 0, 0, 0}, BalanceFactor);
        tree.add("16");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"17", "12", "11", "16", "28", "34"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAddWithNegativeNumbs4() {
        //Creates a tree with 3 height levels and performs different rotation from simpleAddWithNegativeNums2
        tree = new AVL<>(Arrays.asList("-28", "-12", "-34", "-11", "-17"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"-28", "-12", "-11", "-17", "-34"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 0}, getHeights);
        assertArrayEquals(new int[] {1, 0, 0, 0, 0}, BalanceFactor);
        tree.add("-16");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"-17", "-12", "-11", "-16", "-28", "-34"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd5() {
        //Creates a tree with 4 height levels and test different rotations than testAdd3
        tree = new AVL<>(Arrays.asList("22", "06", "28", "05", "11", "23", "32", "04", "09", "13"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(10, tree.size());
        assertArrayEquals(new String[] {"22", "06", "05", "04", "11", "09", "13", "28", "23", "32"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 1, 0, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {1, 0, 1, 0, 0, 0, 0, 0, 0, 0}, BalanceFactor);
        tree.add("10");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"11", "06", "05", "04", "09", "10", "22", "13", "28", "23", "32"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 1, 0, 2, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 1, 0, -1, 0, -1, 0, 0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAddWithNegativeNums5() {
        //Creates a tree with 4 height levels and test different rotations than testAddWithNegativeNums3
        tree = new AVL<>(Arrays.asList("-22", "-06", "-28", "-05", "-11", "-23", "-32", "-04", "-09", "-13"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(10, tree.size());
        assertArrayEquals(new String[] {"-22", "-06", "-05", "-04", "-11", "-09", "-13", "-28", "-23", "-32"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 1, 0, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {1, 0, 1, 0, 0, 0, 0, 0, 0, 0}, BalanceFactor);
        tree.add("-10");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"-11", "-06", "-05", "-04", "-09", "-10", "-22", "-13", "-28", "-23", "-32"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 1, 0, 2, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 1, 0, -1, 0, -1, 0, 0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd6() {
        /** Creates a simple tree with the final resultsult:
         *  Also performs a different rotation than testAdd
         *
         *          1
         *        /   \
         *      0       6
         *
         *
         * **/
        tree = new AVL<>(Arrays.asList("00", "01"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(2, tree.size());
        assertArrayEquals(new String[] {"00", "01"}, getValues);
        assertArrayEquals(new int[] {1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 0}, BalanceFactor);
        tree.add("06");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(3, tree.size());
        assertArrayEquals(new String[] {"01", "00", "06"}, getValues);
        assertArrayEquals(new int[] {1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAddWithNegativeNums6() {
        /** Creates a simple tree with the final resultsult:
         *  Also performs a different rotation than testAdd
         *
         *          -1
         *         /   \
         *      -6       0
         *
         *
         * **/
        tree = new AVL<>(Arrays.asList("00", "-01"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(2, tree.size());
        assertArrayEquals(new String[] {"00", "-01"}, getValues);
        assertArrayEquals(new int[] {1, 0}, getHeights);
        assertArrayEquals(new int[] {1, 0}, BalanceFactor);
        tree.add("-06");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(3, tree.size());
        assertArrayEquals(new String[] {"-06", "-01", "00"}, getValues);
        assertArrayEquals(new int[] {1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd7() {
        //Creates a tree with 2 height levels but performs different rotations than testAdd2 and testAdd4
        tree = new AVL<>(Arrays.asList("60", "12", "78", "63", "90"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"60", "12", "78", "63", "90"}, getValues);
        assertArrayEquals(new int[] {2, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 0, 0, 0, 0}, BalanceFactor);
        tree.add("72");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"63", "60", "12", "78", "72", "90"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 1, 0, 0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAddWithNegativeNums7() {
        //Creates a tree with 2 height levels but performs different rotations than
        // testAddWithNegativeNums2 and testAddWithNegativeNums4
        tree = new AVL<>(Arrays.asList("-60", "-12", "-78", "-63", "-90"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"-60", "-12", "-78", "-63", "-90"}, getValues);
        assertArrayEquals(new int[] {2, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 0, 0, 0, 0}, BalanceFactor);
        tree.add("-72");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"-63", "-60", "-12", "-78", "-72", "-90"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 1, 0, 0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd8() {
        //Creates a tree with 3 height levels with different rotations than other test cases
        tree = new AVL<>(Arrays.asList("30", "14", "36", "32", "40"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"30", "14", "36", "32", "40"}, getValues);
        assertArrayEquals(new int[] {2, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 0, 0, 0, 0}, BalanceFactor);
        tree.add("31");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"32", "30", "14", "31", "36", "40"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testAddWithNegativeNums8() {
        //Creates a tree with 3 height levels with different rotations than other test cases
        tree = new AVL<>(Arrays.asList("-30", "-14", "-36", "-32", "-40"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"-30", "-14", "-36", "-32", "-40"}, getValues);
        assertArrayEquals(new int[] {2, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 0, 0, 0, 0}, BalanceFactor);
        tree.add("-31");
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"-32", "-30", "-14", "-31", "-36", "-40"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveException() {
        //Tests trying to remove a null data / empty tree
        assertThrows(IllegalArgumentException.class, () -> {
            tree.remove(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveException2() {
        //Tests removing data not in a tree
        tree = new AVL<>(Arrays.asList("06", "03", "09", "02", "04", "09", "11", "05", "07", "10", "12", "13"));
        String[] notIn = new String[] {"20", "30"};
        for (String str : notIn) {
            assertThrows(NoSuchElementException.class, () -> {
                tree.remove(str);
            });
        }
    }

    //ALL REMOVE TEST CASES PERFORM ALL POSSIBLE ROTATIONS

    @Test(timeout = TIMEOUT)
    public void testRemove() {
        //Creates a tree and removes one value from it
        tree = new AVL<>(Arrays.asList("16", "21", "08", "07", "18", "23", "25"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"16", "08", "07", "21", "18", "23", "25"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("18"), tree.remove("18"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"16", "08", "07", "23", "21", "25"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 1, 0, 0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveWithNegativeNums() {
        //Creates a tree of negative numbers and removes one value from it
        tree = new AVL<>(Arrays.asList("-16", "-21", "-08", "-07", "-18", "-23", "-25"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"-16", "-08", "-07", "-21", "-18", "-23", "-25"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("-18"), tree.remove("-18"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"-16", "-08", "-07", "-23", "-21", "-25"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 1, 0, 0}, getHeights);
        assertArrayEquals(new int[] {0, 1, 0, 0, 0, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove2() {
        //Creates a medium-sized tree and removes one element from it
        tree = new AVL<>(Arrays.asList("18", "23", "10", "09", "20", "25", "27"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"18", "10", "09", "23", "20", "25", "27"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("09"), tree.remove("09"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"23", "18", "10", "20", "25", "27"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveWithNegativeNums2() {
        //Creates a medium-sized tree of negative numbers and removes one element from it
        tree = new AVL<>(Arrays.asList("-18", "-23", "-10", "-09", "-20", "-25", "-27"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"-18", "-10", "-09", "-23", "-20", "-25", "-27"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("-09"), tree.remove("-09"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"-23", "-18", "-10", "-20", "-25", "-27"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove3() {
        //Creates a medium-sized tree and removes one element from it
        //Performs a different rotation than testRemove2
        tree = new AVL<>(Arrays.asList("15", "20", "07", "06", "17", "22", "24"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"15", "07", "06", "20", "17", "22", "24"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("07"), tree.remove("07"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"20", "15", "06", "17", "22", "24"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveWithNegativeNums3() {
        //Creates a medium-sized tree of negative numbers and removes one element from it
        //Performs a different rotation than testRemoveWithNegativeNums2
        tree = new AVL<>(Arrays.asList("-15", "-20", "-07", "-06", "-17", "-22", "-24"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"-15", "-07", "-06", "-20", "-17", "-22", "-24"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("-07"), tree.remove("-07"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"-20", "-15", "-06", "-17", "-22", "-24"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove4() {
        //Creates a medium-sized tree and removes a single element from it
        //Performs different rotations from testRemove and testRemove2
        tree = new AVL<>(Arrays.asList("09", "14", "01", "05", "11", "16", "18"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"09", "01", "05", "14", "11", "16", "18"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("01"), tree.remove("01"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"14", "09", "05", "11", "16", "18"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveWithNegativeNums4() {
        //Creates a medium-sized tree of negative numbers and removes a single element from it
        //Performs different rotations from testRemoveWithNegativeNums and testRemoveWithNegativeNums2
        tree = new AVL<>(Arrays.asList("-09", "-14", "-01", "-05", "-11", "-16", "-18"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"-09", "-01", "-05", "-14", "-11", "-16", "-18"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("-01"), tree.remove("-01"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"-14", "-09", "-05", "-11", "-16", "-18"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove5() {
        //Creates a medium-sized tree and removes a single element from it
        //Performs a different rotation from testRemove 1, 2, and 4
        tree = new AVL<>(Arrays.asList("10", "15", "02", "01", "12", "17", "19"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"10", "02", "01", "15", "12", "17", "19"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("10"), tree.remove("10"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"15", "02", "01", "12", "17", "19"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveWithNegativeNums5() {
        //Creates a medium-sized tree of negative numbers and removes a single element from it
        //Performs a different rotation from testRemoveWithNegativeNums 1, 2, and 4
        tree = new AVL<>(Arrays.asList("-10", "-15", "-02", "-01", "-12", "-17", "-19"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"-10", "-02", "-01", "-15", "-12", "-17", "-19"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("-10"), tree.remove("-10"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"-15", "-02", "-01", "-12", "-17", "-19"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove6() {
        //Creates a medium-sized tree and removes a single element from it
        //Performs a different rotation than the other testRemove methods
        tree = new AVL<>(Arrays.asList("20", "30", "04", "08", "24", "34", "38"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"20", "04", "08", "30", "24", "34", "38"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("20"), tree.remove("20"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"30", "08", "04", "24", "34", "38"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveWithNegativeNums6() {
        //Creates a medium-sized tree of negative numbers and removes a single element from it
        //Performs a different rotation than the other testRemoveWithNegativeNums methods
        tree = new AVL<>(Arrays.asList("-20", "-30", "-04", "-08", "-24", "-34", "-38"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"-20", "-04", "-08", "-30", "-24", "-34", "-38"}, getValues);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("-20"), tree.remove("-20"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"-30", "-08", "-04", "-24", "-34", "-38"}, getValues);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove7() {
        //Creates a large tree and removes one element from it
        tree = new AVL<>(Arrays.asList("17", "14", "20", "13", "15", "19", "22", "16", "18", "21", "23", "24"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(12, tree.size());
        assertArrayEquals(new String[] {"17", "14", "13", "15", "16", "20", "19", "18", "22", "21", "23", "24"}, getValues);
        assertArrayEquals(new int[] {4, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("13"), tree.remove("13"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"20", "17", "15", "14", "16", "19", "18", "22", "21", "23", "24"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 0, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveWithNegativeNums7() {
        //Creates a large tree of negative numbers and removes one element from it
        tree = new AVL<>(Arrays.asList("-17", "-14", "-20", "-13", "-15", "-19", "-22", "-16", "-18", "-21", "-23", "-24"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(12, tree.size());
        assertArrayEquals(new String[] {"-17", "-14", "-13", "-15", "-16", "-20", "-19", "-18", "-22", "-21", "-23", "-24"}, getValues);
        assertArrayEquals(new int[] {4, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("-13"), tree.remove("-13"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"-20", "-17", "-15", "-14", "-16", "-19", "-18", "-22", "-21", "-23", "-24"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 0, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove8() {
        //Creates a large tree and removes one element from it
        tree = new AVL<>(Arrays.asList("17", "14", "20", "13", "15", "19", "22", "16", "18", "21", "23", "24"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(12, tree.size());
        assertArrayEquals(new String[] {"17", "14", "13", "15", "16", "20", "19", "18", "22", "21", "23", "24"}, getValues);
        assertArrayEquals(new int[] {4, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("17"), tree.remove("17"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"20", "16", "14", "13", "15", "19", "18", "22", "21", "23", "24"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 0, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveWithNegativeNums8() {
        //Creates a large tree of negative numbers and removes one element from it
        tree = new AVL<>(Arrays.asList("-17", "-14", "-20", "-13", "-15", "-19", "-22", "-16", "-18", "-21", "-23", "-24"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(12, tree.size());
        assertArrayEquals(new String[] {"-17", "-14", "-13", "-15", "-16", "-20", "-19", "-18", "-22", "-21", "-23", "-24"}, getValues);
        assertArrayEquals(new int[] {4, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        assertEquals(new String("-17"), tree.remove("-17"));
        preorder();
        getHeights = getHeights();
        BalanceFactor = BalanceFactor();
        getValues = getValues();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"-20", "-16", "-14", "-13", "-15", "-19", "-18", "-22", "-21", "-23", "-24"}, getValues);
        assertArrayEquals(new int[] {3, 2, 1, 0, 0, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {0, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0}, BalanceFactor);
    }

    @Test(timeout = TIMEOUT)
    public void testGetException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.get(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testGetException2() {
        tree = new AVL<>(Arrays.asList("5", "2", "8", "1", "3", "7", "4",
                "6", "9"));
        String[] notIn = new String[] {"10", "11", "12"};
        for (String str : notIn) {
            assertThrows(NoSuchElementException.class, () -> {
                tree.get(str);
            });
        }
    }

    @Test(timeout = TIMEOUT)
    public void testGet() {
        tree = new AVL<>(Arrays.asList("05", "02", "08", "01", "03", "07", "10", "04", "06", "09", "11", "12"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(12, tree.size());
        assertArrayEquals(new String[] {"05", "02", "01", "03", "04", "08", "07", "06", "10", "09", "11", "12"}, getValues);
        assertArrayEquals(new int[] {4, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        String[] elements = new String[] {"05", "02", "08", "01", "03", "07", "10", "04", "06", "09", "11", "12"};
        for (String str : elements) {
            String s = new String(str);
            String results = tree.get(s);
            assertFalse(results == s);
            assertEquals(str, results);
        }
    }

    @Test(timeout = TIMEOUT)
    public void testContainsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.contains(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testContains() {
        tree = new AVL<>(Arrays.asList("05", "02", "08", "01", "03", "07", "10", "04", "06", "09", "11", "12"));
        preorder();
        int[] getHeights = getHeights(), BalanceFactor = BalanceFactor();
        String[] getValues = getValues();
        assertEquals(12, tree.size());
        assertArrayEquals(new String[] {"05", "02", "01", "03", "04", "08", "07", "06", "10", "09", "11", "12"}, getValues);
        assertArrayEquals(new int[] {4, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 0}, getHeights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 1, 0, -1, 0, -1, 0}, BalanceFactor);
        String[] elements = new String[] {"05", "02", "08", "01", "03", "07", "10", "04", "06", "09", "11", "12"};
        for (String str : elements) {
            String s = new String(str);
            assertTrue(tree.contains(s));
        }
        String[] notIn = new String[] {"5000", "1234", "13"};
        for (String str : notIn) {
            String s = new String(str);
            assertFalse(tree.contains(s));
        }
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches1() {
        /*
                          1234
         */
        tree = new AVL<>(Arrays.asList("1234"));
        List<String> expected = Arrays.asList("1234");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);
        tree.clear();
        expected = new ArrayList<>();
        actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches2() {
        /*
                                20
                            /        \
                           10         30
                         /   \      /    \
                        4     14    26    40
                       / \   / \     \    / \
                      2   8 12  16   28  34  50
         */
        tree = new AVL<>(Arrays.asList("20", "10", "30", "04", "14", "26", "40", "02", "08", "12", "16", "28", "34", "50"));
        List<String> expected = Arrays.asList("20", "10", "04", "02", "08", "14", "12", "16", "30", "26", "28", "40", "34", "50");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches3() {
        /*
                                20
                            /        \
                           10         30
                         /   \      /    \
                        4     14    26    40
                       / \   / \     \    / \
                      2   8 12  16   28  34  50
                     /           \            \
                    0             18           60
         */
        tree = new AVL<>(Arrays.asList("20", "10", "30", "04", "14", "26", "40", "02", "08", "12", "16", "28", "34", "50", "00", "18", "60"));
        List<String> expected = Arrays.asList("20", "10", "04", "02", "00", "14", "16", "18", "30", "40", "50", "60");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches4() {
        /*
                                20
                            /        \
                           10         30
                         /   \      /
                        8    14    26
                             / \
                            12  16
         */
        tree = new AVL<>(Arrays.asList("20", "10", "30", "08", "14", "26", "12", "16"));
        List<String> expected = Arrays.asList("20", "10", "14", "12", "16");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches5() {
        /*

                                20
                            /        \
                           10         30
                         /   \      /    \
                        4    14    26     40
                               \     \
                                16    28
         */
        tree = new AVL<>(Arrays.asList("20", "10", "30", "04", "14", "26", "40", "16", "28"));
        List<String> expected = Arrays.asList("20", "10", "14", "16", "30", "26", "28");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches6() {
        /*
                                20
                            /        \
                           10         30
                         /   \      /
                        8    14    26
                             /
                           12
         */
        tree = new AVL<>(Arrays.asList("20", "10", "30", "08", "14", "26", "12"));
        List<String> expected = Arrays.asList("20", "10", "14", "12");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testSortedInBetweenException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.sortedInBetween(null, "001");
        });
    }

    @Test(timeout = TIMEOUT)
    public void testSortedInBetweenException2() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.sortedInBetween("null", null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testSortedInBetweenException3() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.sortedInBetween("001", null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testSortedInBetween() {
        /*
        Test some of the possible combinations that the sortInBetween method could call

                                20
                            /        \
                           10         30
                         /   \      /     \
                        4    14    26      40
                       / \   / \     \    /  \
                      2   8 12  16   28  34   50
                     /           \              \
                    0             18             60
         */
        tree = new AVL<>(Arrays.asList("20", "10", "30", "04", "14", "26", "40", "02", "08", "12", "16", "28", "34", "50", "00", "18", "60"));
        List<String> expected = Arrays.asList("16", "18", "20", "26");
        List<String> actual = tree.sortedInBetween("14", "28");
        assertEquals(expected, actual);
        expected = Arrays.asList("08", "10", "12", "14");
        actual = tree.sortedInBetween("06", "16");
        assertEquals(expected, actual);
        expected = Arrays.asList();
        actual = tree.sortedInBetween("14", "14");
        assertEquals(expected, actual);
        expected = Arrays.asList("02", "04");
        actual = tree.sortedInBetween("00", "06");
        assertEquals(expected, actual);
        expected = Arrays.asList("30", "34", "40", "50");
        actual = tree.sortedInBetween("28", "51");
        assertEquals(expected, actual);
        expected = Arrays.asList("02", "04", "08", "10", "12", "14", "16", "18", "20", "26", "28", "30", "34", "40", "50", "60");
        actual = tree.sortedInBetween("00", "61");
        assertEquals(expected, actual);
    }
}