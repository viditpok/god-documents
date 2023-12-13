import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Your implementation of an AVL.
 *
 * @author Vidit Pokharna
 * @version 1.0
 * @userid vpokharna3
 * @GTID 903772087
 *
 * Collaborators: LIST ALL COLLABORATORS YOU WORKED WITH HERE
 *
 * Resources: LIST ALL NON-COURSE RESOURCES YOU CONSULTED HERE
 */
public class AVL<T extends Comparable<? super T>> {

    // Do not add new instance variables or modify existing ones.
    private AVLNode<T> root;
    private int size;

    /**
     * Constructs a new AVL.
     *
     * This constructor should initialize an empty AVL.
     *
     * Since instance variables are initialized to their default values, there
     * is no need to do anything for this constructor.
     */
    public AVL() {
        // DO NOT IMPLEMENT THIS CONSTRUCTOR!
    }

    /**
     * Constructs a new AVL.
     *
     * This constructor should initialize the AVL with the data in the
     * Collection. The data should be added in the same order it is in the
     * Collection.
     *
     * @param data the data to add to the tree
     * @throws java.lang.IllegalArgumentException if data or any element in data
     *                                            is null
     */
    public AVL(Collection<T> data) {
        if (data == null) {
            throw new IllegalArgumentException("List of data is null so unable to add to tree");
        }
        for (T t : data) {
            if (t == null) {
                throw new IllegalArgumentException("Unable to add null data to tree");
            }
            add(t);
        }
    }

    /**
     * Adds the element to the tree.
     *
     * Start by adding it as a leaf like in a regular BST and then rotate the
     * tree as necessary.
     *
     * If the data is already in the tree, then nothing should be done (the
     * duplicate shouldn't get added, and size should not be incremented).
     *
     * Remember to recalculate heights and balance factors while going back
     * up the tree after adding the element, making sure to rebalance if
     * necessary.
     *
     * Hint: Should you use value equality or reference equality?
     *
     * @param data the data to add
     * @throws java.lang.IllegalArgumentException if data is null
     */
    public void add(T data) {
        if (data == null) {
            throw new IllegalArgumentException("Unable to add null data to tree");
        }
        root = addHelper(root, data);
    }

    /**
     * Helper method used for adding node to AVL
     *
     * @param node node within the tree
     * @param data data to compare with
     * @return root of the tree with data added
     */
    private AVLNode<T> addHelper(AVLNode<T> node, T data) {
        if (node == null) {
            size++;
            return new AVLNode<T>(data);
        }
        int compareValue = node.getData().compareTo(data);
        if (compareValue > 0) {
            node.setLeft(addHelper(node.getLeft(), data));
        } else if (compareValue < 0) {
            node.setRight(addHelper(node.getRight(), data));
        }
        updateHeight(node);
        return balance(node);
    }

    /**
     * Helper method used for setting balance factor and height
     *
     * @param node node to set BF and height
     * @return balance factor of the node being checked
     */
    private void updateHeight(AVLNode<T> node) {
        int leftHeight = -1;
        int rightHeight = -1;
        if (node.getLeft() != null) {
            leftHeight = node.getLeft().getHeight();
        }
        if (node.getRight() != null) {
            rightHeight = node.getRight().getHeight();
        }
        node.setHeight(Math.max(leftHeight, rightHeight) + 1);
        node.setBalanceFactor(leftHeight - rightHeight);
    }

    /**
     * Helper method to rebalance the tree, calling the different rotations when necessary
     *
     * @param node root of the tree
     * @return root of the tree after rebalancing has been done
     */
    private AVLNode<T> balance(AVLNode<T> node) {
        if (node.getBalanceFactor() == -2) {
            if (node.getRight().getBalanceFactor() > 0) {
                node.setRight(rightRotate(node.getRight()));
            }
            node = leftRotate(node);
        } else if (node.getBalanceFactor() == 2) {
            if (node.getLeft().getBalanceFactor() < 0) {
                node.setLeft(leftRotate(node.getLeft()));
            }
            node = rightRotate(node);
        }
        return node;
    }

    /**
     * Left rotation of the tree
     *
     * @param node root of the tree
     * @return node that replaces after rotation
     */
    private AVLNode<T> leftRotate(AVLNode<T> node) {
        AVLNode<T> replace = node.getRight();
        node.setRight(replace.getLeft());
        replace.setLeft(node);
        updateHeight(node);
        updateHeight(replace);
        return replace;
    }

    /**
     * Right rotation of the tree
     *
     * @param node root of the tree
     * @return node that replaces after rotation
     */
    private AVLNode<T> rightRotate(AVLNode<T> node) {
        AVLNode<T> replace = node.getLeft();
        node.setLeft(replace.getRight());
        replace.setRight(node);
        updateHeight(node);
        updateHeight(replace);
        return replace;
    }

    /**
     * Removes and returns the element from the tree matching the given
     * parameter.
     *
     * There are 3 cases to consider:
     * 1: The node containing the data is a leaf (no children). In this case,
     * simply remove it.
     * 2: The node containing the data has one child. In this case, simply
     * replace it with its child.
     * 3: The node containing the data has 2 children. Use the predecessor to
     * replace the data, NOT successor. As a reminder, rotations can occur
     * after removing the predecessor node.
     *
     * Remember to recalculate heights and balance factors while going back
     * up the tree after removing the element, making sure to rebalance if
     * necessary.
     *
     * Do not return the same data that was passed in. Return the data that
     * was stored in the tree.
     *
     * Hint: Should you use value equality or reference equality?
     *
     * @param data the data to remove
     * @return the data that was removed
     * @throws java.lang.IllegalArgumentException if data is null
     * @throws java.util.NoSuchElementException   if the data is not found
     */
    public T remove(T data) {
        if (data == null) {
            throw new IllegalArgumentException("Unable to remove null data to tree");
        }
        AVLNode<T> dummy = new AVLNode<>(null);
        root = removeHelper(root, dummy, data);
        size--;
        return dummy.getData();
    }

    /**
     * Helper method used for removing node from tree
     *
     * @param node node within the tree
     * @param data data to compare with
     * @return data matching parameter in get method
     */
    private AVLNode<T> removeHelper(AVLNode<T> node, AVLNode<T> dummy, T data) {
        if (node == null) {
            throw new NoSuchElementException("The data has not been found");
        }
        if (node.getData().equals(data)) {
            dummy.setData(node.getData());
            if (node.getRight() == null && node.getLeft() == null) {
                return null;
            } else if (node.getRight() == null) {
                return node.getLeft();
            } else if (node.getLeft() == null) {
                return node.getRight();
            } else {
                AVLNode<T> dummy2 = new AVLNode<>(node.getData());
                node.setLeft(rPred(node.getLeft(), dummy2));
                node.setData(dummy2.getData());
            }
        } else if (data.compareTo(node.getData()) < 0) {
            node.setLeft(removeHelper(node.getLeft(), dummy, data));
        } else if (data.compareTo(node.getData()) > 0) {
            node.setRight(removeHelper(node.getRight(), dummy, data));
        }
        updateHeight(node);
        return balance(node);
    }

    /**
     * Helper method used for removing predecessor and placing into removed node
     *
     * @param node node within the tree
     * @param parent data to compare with
     * @return data matching parameter in get method
     */
    private AVLNode<T> rPred(AVLNode<T> node, AVLNode<T> parent) {
        if (node.getRight() == null) {
            parent.setData(node.getData());
            return node.getLeft();
        }
        node.setRight(rPred(node.getRight(), parent));
        updateHeight(node);
        return balance(node);
    }

    /**
     * Returns the element from the tree matching the given parameter.
     *
     * Hint: Should you use value equality or reference equality?
     *
     * Do not return the same data that was passed in. Return the data that
     * was stored in the tree.
     *
     * @param data the data to search for in the tree
     * @return the data in the tree equal to the parameter
     * @throws java.lang.IllegalArgumentException if data is null
     * @throws java.util.NoSuchElementException   if the data is not in the tree
     */
    public T get(T data) {
        if (data == null) {
            throw new IllegalArgumentException("Unable to get null data from tree");
        }
        return getHelper(root, data);
    }

    /**
     * Helper method used for getting node from tree
     *
     * @param node node within the tree
     * @param data data to compare with
     * @return data matching parameter in get method
     */
    private T getHelper(AVLNode<T> node, T data) {
        if (node == null) {
            throw new NoSuchElementException("The data is not in the tree");
        }
        int compareValue = node.getData().compareTo(data);
        if (compareValue > 0) {
            return getHelper(node.getLeft(), data);
        } else if (compareValue < 0) {
            return getHelper(node.getRight(), data);
        } else if (compareValue == 0) {
            return node.getData();
        }
        return node.getData();
    }

    /**
     * Returns whether or not data matching the given parameter is contained
     * within the tree.
     *
     * Hint: Should you use value equality or reference equality?
     *
     * @param data the data to search for in the tree.
     * @return true if the parameter is contained within the tree, false
     * otherwise
     * @throws java.lang.IllegalArgumentException if data is null
     */
    public boolean contains(T data) {
        if (data == null) {
            throw new IllegalArgumentException("Unable to find null data in tree");
        }
        return containsHelper(root, data);
    }

    /**
     * Helper method used for checking if data in some node is in tree
     *
     * @param node node within the tree
     * @param data data to compare with
     * @return whether data is in tree
     */
    private boolean containsHelper(AVLNode<T> node, T data) {
        if (node == null) {
            return false;
        }
        int compareValue = node.getData().compareTo(data);
        if (compareValue > 0) {
            return containsHelper(node.getLeft(), data);
        } else if (compareValue < 0) {
            return containsHelper(node.getRight(), data);
        } else if (compareValue == 0) {
            return true;
        }
        return false;
    }

    /**
     * Returns the height of the root of the tree.
     *
     * Should be O(1).
     *
     * @return the height of the root of the tree, -1 if the tree is empty
     */
    public int height() {
        if (root == null) {
            return -1;
        } else {
            return root.getHeight();
        }
    }

    /**
     * Clears the tree.
     *
     * Clears all data and resets the size.
     */
    public void clear() {
        root = null;
        size = 0;
    }

    /**
     * Returns the data on branches of the tree with the maximum depth. If you
     * encounter multiple branches of maximum depth while traversing, then you
     * should list the remaining data from the left branch first, then the
     * remaining data in the right branch. This is essentially a preorder
     * traversal of the tree, but only of the branches of maximum depth.
     *
     * This must be done recursively.
     *
     * Your list should not have duplicate data, and the data of a branch should be
     * listed in order going from the root to the leaf of that branch.
     *
     * Should run in worst case O(n), but you should not explore branches that
     * do not have maximum depth. You should also not need to traverse branches
     * more than once.
     *
     * Hint: How can you take advantage of the balancing information stored in
     * AVL nodes to discern deep branches?
     *
     * Example Tree:
     *                           10
     *                       /        \
     *                      5          15
     *                    /   \      /    \
     *                   2     7    13    20
     *                  / \   / \     \  / \
     *                 1   4 6   8   14 17  25
     *                /           \          \
     *               0             9         30
     *
     * Returns: [10, 5, 2, 1, 0, 7, 8, 9, 15, 20, 25, 30]
     *
     * @return the list of data in branches of maximum depth in preorder
     * traversal order
     */
    public List<T> deepestBranches() {
        List<T> list = new ArrayList<T>();
        rDeepBranch(root, list);
        return list;
    }

    /**
     * Recursive method to traverse the avl tree
     *
     * @param node the node that the recursive method will take to traverse the bst
     * @param list list that will be added to
     */
    private void rDeepBranch(AVLNode<T> node, List<T> list) {
        if (node == null) {
            return;
        } else {
            list.add(node.getData());
            if (node.getLeft() != null) {
                int difference = node.getHeight() - node.getLeft().getHeight();
                if (difference == 1 || difference == 0) {
                    rDeepBranch(node.getLeft(), list);
                }
            }
            if (node.getRight() != null) {
                int difference = node.getHeight() - node.getRight().getHeight();
                if (difference == 1 || difference == 0) {
                    rDeepBranch(node.getRight(), list);
                }
            }
        }
    }

    /**
     * Returns a sorted list of data that are within the threshold bounds of
     * data1 and data2. That is, the data should be > data1 and < data2.
     *
     * This must be done recursively.
     *
     * Should run in worst case O(n), but this is heavily dependent on the
     * threshold data. You should not explore branches of the tree that do not
     * satisfy the threshold.
     *
     * Example Tree:
     *                           10
     *                       /        \
     *                      5          15
     *                    /   \      /    \
     *                   2     7    13    20
     *                  / \   / \     \  / \
     *                 1   4 6   8   14 17  25
     *                /           \          \
     *               0             9         30
     *
     * sortedInBetween(7, 14) returns [8, 9, 10, 13]
     * sortedInBetween(3, 8) returns [4, 5, 6, 7]
     * sortedInBetween(8, 8) returns []
     *
     * @param data1 the smaller data in the threshold
     * @param data2 the larger data in the threshold
     * @return a sorted list of data that is > data1 and < data2
     * @throws IllegalArgumentException if data1 or data2 are null
     * or if data1 > data2
     */
    public List<T> sortedInBetween(T data1, T data2) {
        if (data1 == null || data2 == null) {
            throw new IllegalArgumentException("The data given is null");
        } else if (data1.compareTo(data2) > 0) {
            throw new IllegalArgumentException("1st data input is greater than 2nd data input");
        }
        List<T> list = new ArrayList<T>();
        rSortBetween(root, list, data1, data2);
        return list;
    }

    /**
     * Recursive method to traverse the avl tree
     *
     * @param curr the node that the recursive method will take to traverse the bst
     * @param list the list of nodes forming the preorder traversal
     * @param data1 lower bound
     * @param data2 upper bound
     */
    private void rSortBetween(AVLNode<T> curr, List<T> list, T data1, T data2) {
        if (curr != null) {
            rSortBetween(curr.getLeft(), list, data1, data2);
            if (curr.getData().compareTo(data1) > 0 && curr.getData().compareTo(data2) < 0) {
                list.add(curr.getData());
            }
            rSortBetween(curr.getRight(), list, data1, data2);
        }
    }

    /**
     * Returns the root of the tree.
     *
     * For grading purposes only. You shouldn't need to use this method since
     * you have direct access to the variable.
     *
     * @return the root of the tree
     */
    public AVLNode<T> getRoot() {
        // DO NOT MODIFY THIS METHOD!
        return root;
    }

    /**
     * Returns the size of the tree.
     *
     * For grading purposes only. You shouldn't need to use this method since
     * you have direct access to the variable.
     *
     * @return the size of the tree
     */
    public int size() {
        // DO NOT MODIFY THIS METHOD!
        return size;
    }
}
