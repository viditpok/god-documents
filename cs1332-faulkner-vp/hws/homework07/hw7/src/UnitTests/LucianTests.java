import org.junit.Test;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;

import static org.junit.Assert.*;

public class LucianTests {

    private static final int TIMEOUT = 200;
    private AVL<Integer> tree;
    private List<Integer> values;
    private List<Integer> balanceFactors;
    private List<Integer> heights;

    private void print() {
        levelorder();
        System.out.println("\nNode:\t" + values);
        System.out.println("BF:\t\t" + balanceFactors);
        System.out.println("Height:\t" + heights);
    }

    private void levelorder() {
        values = new ArrayList<>();
        balanceFactors = new ArrayList<>();
        heights = new ArrayList<>();
        AVLNode<Integer> root = tree.getRoot();
        Queue<AVLNode<Integer>> q = new LinkedBlockingQueue<>();
        // Return empty list if tree is empty
        if (root == null) {
            return;
        }
        q.add(root);
        // Add to list in level order using queue
        while (q.size() > 0) {
            AVLNode<Integer> curr = q.remove();
            if (curr != null) {
                values.add(curr.getData());
                heights.add(curr.getHeight());
                balanceFactors.add(curr.getBalanceFactor());
                if (curr.getLeft() != null) {
                    q.add(curr.getLeft());
                }
                if (curr.getRight() != null) {
                    q.add(curr.getRight());
                }
            }
        }
    }

    private Object[] levelorderValues() {
        levelorder();
        return values.toArray();
    }

    private Object[] levelorderHeights() {
        levelorder();
        return heights.toArray();
    }

    private Object[] levelorderBalanceFactors() {
        levelorder();
        return balanceFactors.toArray();
    }

    @Test(timeout = TIMEOUT)
    public void testDefaultConstructor() {
        tree = new AVL<>();
        assertEquals(0, tree.size());
        print();
        assertEquals(null, tree.getRoot());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testConstructorNullListException() {
        tree = new AVL<>(null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testConstructorNullDataException() {
        List<Integer> list = new ArrayList<>();
        list.add(0);
        list.add(null);
        list.add(1);
        tree = new AVL<>(list);
    }

    @Test(timeout = TIMEOUT)
    public void testConstructorEmptyList() {
        List<Integer> list = new ArrayList<>();
        tree = new AVL<>(list);
        assertEquals(0, tree.size());
        print();
        assertEquals(null, tree.getRoot());
    }

    @Test(timeout = TIMEOUT)
    public void testConstructorOneNode() {
        List<Integer> list = new ArrayList<>();
        list.add(1);
        tree = new AVL<>(list);
        assertEquals(1, tree.size());
        print();
        assertEquals(Integer.valueOf(1), tree.getRoot().getData());
    }

    @Test(timeout = TIMEOUT)
    public void testConstructor() {
        // Has rebalancing
        List<Integer> list = new ArrayList<>();
        list.add(66);
        list.add(21);
        list.add(34);
        list.add(71);
        list.add(16);
        list.add(10);
        list.add(12);
        list.add(25);
        list.add(5);
        list.add(11);
        tree = new AVL<>(list);
        print();
        assertEquals(10, tree.size());
        assertArrayEquals(new Integer[]{16, 10, 34, 5, 12, 21, 66, 11, 25, 71}, levelorderValues());
        assertArrayEquals(new Integer[]{0, -1, 0, 0, 1, -1, -1, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{3, 2, 2, 0, 1, 1, 1, 0, 0, 0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testConstructorDuplicates() {
        List<Integer> list = new ArrayList<>();
        list.add(1);
        list.add(2);
        list.add(2);
        list.add(3);
        tree = new AVL<>(list);
        print();
        assertEquals(3, tree.size());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testAddNullDataException() {
        tree = new AVL<>();
        tree.add(null);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd() {
        tree = new AVL<>();

        // First element
        tree.add(5);
        assertEquals(1, tree.size());
        print();
        assertArrayEquals(new Integer[]{5}, levelorderValues());
        assertArrayEquals(new Integer[]{0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{0}, levelorderHeights());

        // Duplicate data
        tree.add(5);
        assertEquals(1, tree.size());
        print();
        assertArrayEquals(new Integer[]{5}, levelorderValues());
        assertArrayEquals(new Integer[]{0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{0}, levelorderHeights());

        // Add without rotations
        tree.add(3);
        tree.add(7);
        tree.add(1);
        tree.add(4);
        tree.add(6);
        tree.add(8);
        tree.add(9);
        tree.add(0);
        assertEquals(9, tree.size());
        print();
        assertArrayEquals(new Integer[]{5,3,7,1,4,6,8,0,9}, levelorderValues());
        assertArrayEquals(new Integer[]{0,1,-1,1,0,0,-1,0,0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{3,2,2,1,0,0,1,0,0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testAddLeftRotation() {
        tree = new AVL<>();

        /* Add with left rotation
                1
                 \              2
                  2    -->    /   \
                   \         1     3
                    3
         */
        tree.add(1);
        tree.add(2);
        tree.add(3);
        assertEquals(3, tree.size());
        print();
        assertArrayEquals(new Integer[]{2,1,3}, levelorderValues());
        assertArrayEquals(new Integer[]{0,0,0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{1,0,0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testAddRightRotation() {
        tree = new AVL<>();

        /* Add with right rotation
                    3
                   /           2
                  2    -->   /   \
                 /          1     3
                1
         */
        tree.add(3);
        tree.add(2);
        tree.add(1);
        assertEquals(3, tree.size());
        print();
        assertArrayEquals(new Integer[]{2,1,3}, levelorderValues());
        assertArrayEquals(new Integer[]{0,0,0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{1,0,0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testAddRightLeftRotation() {
        tree = new AVL<>();

        /* Add with right-left rotation
                1
                 \             2
                  3    -->   /   \
                 /          1     3
                2
         */
        tree.add(1);
        tree.add(3);
        tree.add(2);
        assertEquals(3, tree.size());
        print();
        assertArrayEquals(new Integer[]{2,1,3}, levelorderValues());
        assertArrayEquals(new Integer[]{0,0,0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{1,0,0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testAddLeftRightRotation() {
        tree = new AVL<>();

        /* Add with left-right rotation
                3
               /               2
              1     -->      /   \
               \            1     3
                2
         */
        tree.add(3);
        tree.add(1);
        tree.add(2);
        assertEquals(3, tree.size());
        print();
        assertArrayEquals(new Integer[]{2,1,3}, levelorderValues());
        assertArrayEquals(new Integer[]{0,0,0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{1,0,0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveRoot() {
        // Remove root, no rebalance
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9));
        assertEquals(Integer.valueOf(4), tree.remove(4));
        assertEquals(8, tree.size());
        print();
        assertArrayEquals(new Integer[]{3, 2, 6, 1, 5, 8, 7, 9}, levelorderValues());
        assertArrayEquals(new Integer[]{-1, 1, -1, 0, 0, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{3, 1, 2, 0, 0, 1, 0, 0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveLeaf() {
        // Remove leaf, no rebalance
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9));
        assertEquals(Integer.valueOf(7), tree.remove(7));
        assertEquals(8, tree.size());
        print();
        assertArrayEquals(new Integer[]{4, 2, 6, 1, 3, 5, 8, 9}, levelorderValues());
        assertArrayEquals(new Integer[]{-1, 0, -1, 0, 0, 0, -1, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{3, 1, 2, 0, 0, 0, 1, 0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveLeafRebalance() {
        // Remove leaf, rebalance on node 6
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9));
        assertEquals(Integer.valueOf(5), tree.remove(5));
        assertEquals(8, tree.size());
        print();
        assertArrayEquals(new Integer[]{4, 2, 8, 1, 3, 6, 9, 7}, levelorderValues());
        assertArrayEquals(new Integer[]{-1, 0, 1, 0, 0, -1, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{3, 1, 2, 0, 0, 1, 0, 0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveInternal() {
        // Remove internal node (1 child), no rebalance
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9));
        assertEquals(Integer.valueOf(2), tree.remove(2));
        assertEquals(8, tree.size());
        print();
        assertArrayEquals(new Integer[]{4, 1, 6, 3, 5, 8, 7, 9}, levelorderValues());
        assertArrayEquals(new Integer[]{-1, -1, -1, 0, 0, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{3, 1, 2, 0, 0, 1, 0, 0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveRootRebalance1() {
        // Remove root (2 children), rebalance on predecessor node (1)
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9));
        assertEquals(Integer.valueOf(1), tree.remove(1));
        assertEquals(Integer.valueOf(4), tree.remove(4));
        assertEquals(7, tree.size());
        print();
        assertArrayEquals(new Integer[]{6, 3, 8, 2, 5, 7, 9}, levelorderValues());
        assertArrayEquals(new Integer[]{0, 0, 0, 0, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{2, 1, 1, 0, 0, 0, 0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveRootRebalance2() {
        // Remove root, rebalance on predecessor node (3)
        tree = new AVL<>(Arrays.asList(6, 3, 8, 9));
        assertEquals(Integer.valueOf(6), tree.remove(6));
        assertEquals(3, tree.size());
        print();
        assertArrayEquals(new Integer[]{8, 3, 9}, levelorderValues());
        assertArrayEquals(new Integer[]{0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{1, 0, 0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveInternalRebalance() {
        // Remove root, rebalance on predecessor node (3)
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9));
        assertEquals(Integer.valueOf(1), tree.remove(1));
        assertEquals(Integer.valueOf(2), tree.remove(2));
        assertEquals(7, tree.size());
        print();
        assertArrayEquals(new Integer[]{6, 4, 8, 3, 5, 7, 9}, levelorderValues());
        assertArrayEquals(new Integer[]{0, 0, 0, 0, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{2, 1, 1, 0, 0, 0, 0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveInternal2() {
        // Remove internal nodes, no rebalancing
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9));
        assertEquals(Integer.valueOf(6), tree.remove(6));
        assertEquals(Integer.valueOf(7), tree.remove(7));
        assertEquals(Integer.valueOf(8), tree.remove(8));
        assertEquals(Integer.valueOf(9), tree.remove(9));
        assertEquals(5, tree.size());
        print();
        assertArrayEquals(new Integer[]{4, 2, 5, 1, 3}, levelorderValues());
        assertArrayEquals(new Integer[]{1, 0, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{2, 1, 0, 0, 0}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testRemoveNullException() {
        // Attempt to remove null data
        tree = new AVL<>(Arrays.asList(6, 3, 8, 9));
        tree.remove(null);
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testRemoveNotFoundException() {
        // Attempt to remove data not in tree
        tree = new AVL<>(Arrays.asList(6, 3, 8, 9));
        tree.remove(4);
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testRemoveEmptyTreeException() {
        // Attempt to remove from empty tree
        tree = new AVL<>();
        tree.remove(1);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveLastElement() {
        // Remove last element from tree
        tree = new AVL<>(Arrays.asList(6));
        assertEquals(Integer.valueOf(6), tree.remove(6));
        assertEquals(0, tree.size());
        print();
        assertArrayEquals(new Integer[]{}, levelorderValues());
        assertArrayEquals(new Integer[]{}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{}, levelorderHeights());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveFromAbsurdlyLargeTree() {
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20));
        assertEquals(20, tree.size());
        print();
        assertArrayEquals(new Integer[]{8, 4, 16, 2, 6, 12, 18, 1, 3, 5, 7, 10, 14, 17, 19, 9, 11, 13, 15, 20}, levelorderValues());
        assertArrayEquals(new Integer[]{-1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{4, 2, 3, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0}, levelorderHeights());

        // Remove internal node 6 with two children
        assertEquals(Integer.valueOf(6), tree.remove(6));
        assertEquals(19, tree.size());
        print();
        assertArrayEquals(new Integer[]{8, 4, 16, 2, 5, 12, 18, 1, 3, 7, 10, 14, 17, 19, 9, 11, 13, 15, 20}, levelorderValues());
        assertArrayEquals(new Integer[]{-1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{4, 2, 3, 1, 1, 2, 2, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0}, levelorderHeights());

        // Remove root 8
        assertEquals(Integer.valueOf(8), tree.remove(8));
        assertEquals(18, tree.size());
        print();
        assertArrayEquals(new Integer[]{7, 4, 16, 2, 5, 12, 18, 1, 3, 10, 14, 17, 19, 9, 11, 13, 15, 20}, levelorderValues());
        assertArrayEquals(new Integer[]{-1, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{4, 2, 3, 1, 0, 2, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0}, levelorderHeights());

        // Remove root 7
        assertEquals(Integer.valueOf(7), tree.remove(7));
        assertEquals(17, tree.size());
        print();
        assertArrayEquals(new Integer[]{5, 2, 16, 1, 4, 12, 18, 3, 10, 14, 17, 19, 9, 11, 13, 15, 20}, levelorderValues());
        assertArrayEquals(new Integer[]{-1, -1, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{4, 2, 3, 0, 1, 2, 2, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0}, levelorderHeights());

        // Remove internal node 16
        assertEquals(Integer.valueOf(16), tree.remove(16));
        assertEquals(16, tree.size());
        print();
        assertArrayEquals(new Integer[]{5, 2, 15, 1, 4, 12, 18, 3, 10, 14, 17, 19, 9, 11, 13, 20}, levelorderValues());
        assertArrayEquals(new Integer[]{-1, -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, -1, 0, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{4, 2, 3, 0, 1, 2, 2, 0, 1, 1, 0, 1, 0, 0, 0, 0}, levelorderHeights());

        // Remove leaf node 3
        assertEquals(Integer.valueOf(3), tree.remove(3));
        assertEquals(15, tree.size());
        print();
        assertArrayEquals(new Integer[]{15, 5, 18, 2, 12, 17, 19, 1, 4, 10, 14, 20, 9, 11, 13}, levelorderValues());
        assertArrayEquals(new Integer[]{1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0}, levelorderBalanceFactors());
        assertArrayEquals(new Integer[]{4, 3, 2, 1, 2, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0}, levelorderHeights());

        // Remove the rest of the nodes
        ArrayList<Integer> remainingNodes = new ArrayList<>(Arrays.asList(new Integer[]{15, 5, 18, 2, 12, 17, 19, 1, 4, 10, 14, 20, 9, 11, 13}));
        for (int size = tree.size() - 1; size > 0; size--) {
            Integer nodeToRemove = remainingNodes.get((int) (Math.random() * remainingNodes.size()));
            remainingNodes.remove(nodeToRemove);
            assertEquals(nodeToRemove, tree.remove(nodeToRemove));
            assertEquals(size, tree.size());
        }
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testSearchNullLowerBoundException() {
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20));
        tree.sortedInBetween(null, 0);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testSearchNullUpperBoundException() {
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20));
        tree.sortedInBetween(0, null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testSearchInvalidBoundsException() {
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20));
        tree.sortedInBetween(5, 4);
    }

    @Test(timeout = TIMEOUT)
    public void testSearchWholeTree() {
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20));
        print();
        assertArrayEquals(new Integer[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},tree.sortedInBetween(0, 100).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testSearchSameBounds() {
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20));
        print();
        assertArrayEquals(new Integer[]{},tree.sortedInBetween(5, 5).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testSearchNoneFound() {
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20));
        print();
        assertArrayEquals(new Integer[]{},tree.sortedInBetween(20, 25).toArray());
        assertArrayEquals(new Integer[]{},tree.sortedInBetween(-1, 0).toArray());
        assertArrayEquals(new Integer[]{},tree.sortedInBetween(10, 14).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testSearch() {
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20));
        print();
        assertArrayEquals(new Integer[]{2,3},tree.sortedInBetween(1,4).toArray());
        assertArrayEquals(new Integer[]{11,12,13,14,15,16,17},tree.sortedInBetween(10,18).toArray());
        assertArrayEquals(new Integer[]{17},tree.sortedInBetween(16,18).toArray());
        assertArrayEquals(new Integer[]{20},tree.sortedInBetween(19,100).toArray());
        assertArrayEquals(new Integer[]{1,2},tree.sortedInBetween(-100,3).toArray());
        assertArrayEquals(new Integer[]{4,5,6,7,8,9,10,11,12,13,14,15},tree.sortedInBetween(3,16).toArray());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testGetNullException() {
        // Attempt to get null data
        tree = new AVL<>(Arrays.asList(6, 3, 8, 9));
        tree.get(null);
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testGetNotFoundException() {
        // Attempt to get data not in tree
        tree = new AVL<>(Arrays.asList(6, 3, 8, 9));
        tree.get(4);
    }

    @Test(timeout = TIMEOUT)
    public void testGet() {
        // Get data from each node in 20-node tree
        tree = new AVL<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20));
        print();
        for (int i = 1; i <= 20; i++) {
            assertEquals(Integer.valueOf(i), tree.get(i));
        }
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testContainsNullException() {
        // Attempt to get null data
        tree = new AVL<>(Arrays.asList(6, 3, 8, 9));
        tree.contains(null);
    }

    @Test(timeout = TIMEOUT)
    public void testContains() {
        // Test contains for data in range 2-40, only evens are in tree
        tree = new AVL<>(Arrays.asList(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40));
        print();
        for (int i = 2; i <= 40; i++) {
            if (i % 2 == 0)
                assertTrue(tree.contains(i));
            else
                assertFalse(tree.contains(i));
        }
    }

    @Test(timeout = TIMEOUT)
    public void testHeight() {
        // Height = -1 (empty)
        tree = new AVL<>();
        print();
        assertEquals(-1, tree.height());

        // Height = 0
        tree = new AVL<>(Arrays.asList(2));
        print();
        assertEquals(0, tree.height());

        // Height = 1
        tree = new AVL<>(Arrays.asList(2, 4));
        print();
        assertEquals(1, tree.height());

        // Height = 2
        tree = new AVL<>(Arrays.asList(2, 4, 6, 8, 10));
        print();
        assertEquals(2, tree.height());

        // Height = 3
        tree = new AVL<>(Arrays.asList(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22));
        print();
        assertEquals(3, tree.height());

        // Height = 4
        tree = new AVL<>(Arrays.asList(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40));
        print();
        assertEquals(4, tree.height());
    }

    @Test(timeout = TIMEOUT)
    public void testClear() {
        tree = new AVL<>(Arrays.asList(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40));
        tree.clear();
        print();
        assertNull(tree.getRoot());
        assertEquals(0, tree.size());
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches() {
        // Empty tree
        tree = new AVL<>();
        print();
        assertArrayEquals(new Integer[]{}, tree.deepestBranches().toArray());

        // 1 branch, depth 1
        tree.add(1);
        print();
        assertArrayEquals(new Integer[]{1}, tree.deepestBranches().toArray());

        // 1 branch, depth 2
        tree.add(2);
        print();
        assertArrayEquals(new Integer[]{1, 2}, tree.deepestBranches().toArray());

        // 2 branches, depth 2
        tree.add(3);
        print();
        assertArrayEquals(new Integer[]{2, 1, 3}, tree.deepestBranches().toArray());

        // 1 branch, depth 3
        tree.add(4);
        print();
        assertArrayEquals(new Integer[]{2, 3, 4}, tree.deepestBranches().toArray());

        // 2 branches, depth 3
        tree.add(5);
        print();
        assertArrayEquals(new Integer[]{2, 4, 3, 5}, tree.deepestBranches().toArray());

        // 3 branches, depth 3
        tree.add(6);
        print();
        assertArrayEquals(new Integer[]{4, 2 , 1, 3, 5, 6}, tree.deepestBranches().toArray());

        // 4 branches, depth 3
        tree.add(7);
        print();
        assertArrayEquals(new Integer[]{4, 2, 1, 3, 6, 5, 7}, tree.deepestBranches().toArray());

        // 1 branches, depth 4
        tree.add(8);
        print();
        assertArrayEquals(new Integer[]{4, 6, 7, 8}, tree.deepestBranches().toArray());

        // 2 branches, depth 4
        tree.add(9);
        print();
        assertArrayEquals(new Integer[]{4, 6, 8, 7, 9}, tree.deepestBranches().toArray());

        // 3 branches, depth 4
        tree.add(10);
        print();
        assertArrayEquals(new Integer[]{4, 8, 6, 5, 7, 9, 10}, tree.deepestBranches().toArray());

        // 4 branches, depth 4
        tree.add(11);
        print();
        assertArrayEquals(new Integer[]{4, 8, 6, 5, 7, 10, 9, 11}, tree.deepestBranches().toArray());

        // 5 branches, depth 4
        tree.add(12);
        print();
        assertArrayEquals(new Integer[]{8, 4, 2, 1, 3, 6, 5, 7, 10, 11, 12}, tree.deepestBranches().toArray());

        // 6 branches, depth 4
        tree.add(13);
        print();
        assertArrayEquals(new Integer[]{8, 4, 2, 1, 3, 6, 5, 7, 10, 12, 11, 13}, tree.deepestBranches().toArray());

        // 7 branches, depth 4
        tree.add(14);
        print();
        assertArrayEquals(new Integer[]{8, 4, 2, 1, 3, 6, 5, 7, 12, 10, 9, 11, 13, 14}, tree.deepestBranches().toArray());

        // 8 branches, depth 4
        tree.add(15);
        print();
        assertArrayEquals(new Integer[]{8, 4, 2, 1, 3, 6, 5, 7, 12, 10, 9, 11, 14, 13, 15}, tree.deepestBranches().toArray());

        // 1 branches, depth 5
        tree.add(16);
        print();
        assertArrayEquals(new Integer[]{8, 12, 14, 15, 16}, tree.deepestBranches().toArray());

        // 2 branches, depth 5
        tree.add(17);
        print();
        assertArrayEquals(new Integer[]{8, 12, 14, 16, 15, 17}, tree.deepestBranches().toArray());

        // 3 branches, depth 5
        tree.add(18);
        print();
        assertArrayEquals(new Integer[]{8, 12, 16, 14, 13, 15, 17, 18}, tree.deepestBranches().toArray());

        // 4 branches, depth 5
        tree.add(19);
        print();
        assertArrayEquals(new Integer[]{8, 12, 16, 14, 13, 15, 18, 17, 19}, tree.deepestBranches().toArray());

        // 5 branches, depth 5
        tree.add(20);
        print();
        assertArrayEquals(new Integer[]{8, 16, 12, 10, 9, 11, 14, 13, 15, 18, 19, 20}, tree.deepestBranches().toArray());
    }
}