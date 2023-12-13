/**
 * CS 2110 Summer 2022 HW1
 * Part 1 - Coding a bit vector
 *
 * @author Vidit Pokharna
 *
 * Global rules for this file:
 * - You may not use multiplication, division or modulus in any method.
 * - You may not use more than 2 conditionals per method, and you may only use
 *   them in select methods. Conditionals are if-statements, if-else statements,
 *   or ternary expressions. The else block associated with an if-statement does
 *   not count toward this sum.
 * - You may not use looping constructs in most methods. Looping constructs
 *   include for loops, while loops and do-while loops. See below for exceptions
 * - In those methods that allow looping, you may not use more than 2 looping
 *   constructs, and they may not be nested.
 * - You may not declare any file-level variables.
 * - You may not declare any objects.
 * - You may not use switch statements.
 * - You may not use casting.
 * - You may not use the unsigned right shift operator (>>>)
 * - You may not write any helper methods, or call any method from this or
 *   another file to implement any method. Recursive solutions are not
 *   permitted.
 * - You may only perform addition or subtraction with the number 1.
 *
 * Method-specific rules for this file:
 * - All methods must be done in one line, except for: zerosCount, onesCount,
 *   and size.
 * - Looping and conditionals as described above are only allowed for:
 *   zerosCount, onesCount, and size
 * - You can only use addition and subtraction in: zerosCount, onesCount, and
 *   size
 */
public class BitVector
{
    /**
     * 32-bit data initialized to all zeros. Here is what you will be using to
     * represent the Bit Vector. Do not change its scope from private.
     */
    private int bits;

    /** You may not add any more fields to this class other than the given one. */

    /**
     * Sets the bit (sets to 1) pointed to by index.
     * @param index index of which bit to set.
     *              0 for the least significant bit (right most bit).
     *              31 for the most significant bit.
     */
    public void set(int index)
    {
        bits = ((1 << index) | (bits));
    }

    /**
     * Clears the bit (sets to 0) pointed to by index.
     * @param index index of which bit to set.
     *              0 for the least significant bit (right most bit).
     *              31 for the most significant bit.
     */
    public void clear(int index)
    {
        bits = (~(1 << index) & (bits));
    }

    /**
     * Toggles the bit (sets to the opposite of its current value) pointed to by
     * index.
     * @param index index of which bit to set.
     *              0 for the least significant bit (right most bit).
     *              31 for the most significant bit.
     */
    public void toggle(int index)
    {
        bits = ((1 << index) ^ (bits));
    }

    /**
     * Returns true if the bit pointed to by index is currently set.
     * @param index index of which bit to check.
     *              0 for the least significant bit (right-most bit).
     *              31 for the most significant bit.
     * @return true if the bit is set, false if the bit is clear.
     *         If the index is out of range (index >= 32), then return false.
     */
    public boolean isSet(int index)
    {
        return (((bits & (1 << index)) != 0) && (index < 32));
    }

    /**
     * Returns true if the bit pointed to by index is currently clear.
     * @param index index of which bit to check.
     *              0 for the least significant bit (right-most bit).
     *              31 for the most significant bit.
     * @return true if the bit is clear, false if the bit is set.
     *         If the index is out of range (index >= 32), then return true.
     */
    public boolean isClear(int index)
    {
        return (((bits & (1 << index)) == 0) || (index >= 32));
    }

    /**
     * Returns the number of bits currently set (=1) in this bit vector.
     * You may use the ++ operator to increment your counter.
     */
    public int onesCount()
    {
        int count = 0;
        while (bits != 0) {
            bits &= (bits - 1);
            count++;
        }
        return count;
    }

    /**
     * Returns the number of bits currently clear (=0) in this bit vector.
     * You may use the ++ operator to increment your counter.
     */
    public int zerosCount()
    {
        bits = ~bits;
        int count = 0;
        while (bits != 0) {
            bits &= (bits - 1);
            count++;
        }
        return count;
    }

    /**
     * Returns the "size" of this BitVector. The size of this bit vector is
     * defined to be the minimum number of bits that will represent all of the
     * ones.
     *
     * For example, the size of the bit vector 00010000 will be 5.
     */
    public int size()
    {
        int count = 0;
        if (bits == 0) {
            return 1;
        } else if (bits > 0) {
            while (bits != 0) {
                count++;
                bits >>= 1;
            }
        } else {
            while (bits != 0) {
                count++;
                bits <<= 1;
            }
        }
        return count;
    }
}