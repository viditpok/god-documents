import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;


import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertArrayEquals;

/**
 * JUnits!
 * @author Adhish Rajan
 * @version 1.0
 */
public class PatternMatchingExtraTests {

    private static final int TIMEOUT = 200;

    private String kmpPattern;
    private String kmpPattern2;
    private String kmpText;
    private String kmpText2;
    private String kmpNoMatch;
    private List<Integer> kmpAnswer;
    private List<Integer> kmpPatternEquivalencyAnswer;

    private String noMatchPattern;
    private String noMatchText;
    

    private String sellPattern;
    private String sellText;
    private String sellNoMatch;
    private List<Integer> sellAnswer;

    private String multiplePattern;
    private String multipleText;
    private List<Integer> multipleAnswer;

    private List<Integer> emptyList;

    private CharacterComparator comparator;


    @Before
    public void setUp() {
        kmpPattern = "aaba";
        kmpText = "aabaacaadaabaaba";
        kmpNoMatch = "ababbaba";

        kmpPattern2 = "nano";
        kmpText2 = "banananobano";


        noMatchPattern = "12345";
        noMatchText = "123514231425342";


        kmpAnswer = new ArrayList<>();
        kmpAnswer.add(0);
        kmpAnswer.add(9);
        kmpAnswer.add(12);

        kmpPatternEquivalencyAnswer = new ArrayList<>();
        kmpPatternEquivalencyAnswer.add(0);

        sellPattern = "sell";
        sellText = "She sells seashells by the seashore.";
        sellNoMatch = "sea lions trains cardinal boardwalk";

        sellAnswer = new ArrayList<>();
        sellAnswer.add(4);

        multiplePattern = "ab";
        multipleText = "abab";


        emptyList = new ArrayList<>();

        comparator = new CharacterComparator();
    }

    @Test(timeout = TIMEOUT)
    public void exceptionsKMP() {
        assertThrows(IllegalArgumentException.class, () -> {
                PatternMatching.kmp("null", "null", null);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.kmp("", "null", comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.kmp("null", null, comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
                PatternMatching.kmp(null, "null", comparator);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable1() {
        /*
            text: aabaacaadaabaaba
            pattern: aaba
            failure table: [0, 1, 0, 1]
            comparisons: 4
         */
        
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 1, 0, 1};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 4.", 4, comparator.getComparisonCount());
    }


    @Test(timeout = TIMEOUT)
    public void testKMPMatch1() {
        /*
            pattern: aaba
            text: aabaacaadaabaaba
            indices: 0, 9, 12
            expected total comparison: 24

            failure table: [0, 0, 1, 2, 3]
            comparisons: 4

        a | a | b | a | a | c | a | a | d | a | a | b | a | a | b | a |
        --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
        a | a | b | a |   |   |   |   |   |   |   |   |   |   |   |   |    
        - | - | - | - |   |   |   |   |   |   |   |   |   |   |   |   |   comparisons: 4
          |   |   | a | a | b | a |   |   |   |   |   |   |   |   |   |
          |   |   |   | - | - |   |   |   |   |   |   |   |   |   |   |   comparisons: 2
          |   |   |   | a | a | b | a |   |   |   |   |   |   |   |   |
          |   |   |   |   | - |   |   |   |   |   |   |   |   |   |   |   comparisons: 1
          |   |   |   |   | a | a | b | a |   |   |   |   |   |   |   |
          |   |   |   |   | - |   |   |   |   |   |   |   |   |   |   |   comparisons: 1
          |   |   |   |   |   | a | a | b | a |   |   |   |   |   |   |   
          |   |   |   |   |   | - | - | - |   |   |   |   |   |   |   |   comparisons: 3
          |   |   |   |   |   |   | a | a | b | a |   |   |   |   |   |      
          |   |   |   |   |   |   |   | - |   |   |   |   |   |   |   |   comparisons: 1
          |   |   |   |   |   |   |   | a | a | b | a |   |   |   |   |       
          |   |   |   |   |   |   |   | - |   |   |   |   |   |   |   |   comparisons: 1
          |   |   |   |   |   |   |   |   | a | a | b | a |   |   |   |       
          |   |   |   |   |   |   |   |   | - | - | - | - |   |   |   |   comparisons: 4
          |   |   |   |   |   |   |   |   |   |   |   | a | a | b | a |       
          |   |   |   |   |   |   |   |   |   |   |   |   | - | - | - |   comparisons: 4


         comparisons: 20
         */
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 24.", 24, comparator.getComparisonCount());
    }


    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable2() {
        /*
            text: banananobano
            pattern: nano
            failure table: [0, 0, 1, 0]
            comparisons: 4
         */
        
        int[] failureTable2 = PatternMatching
                .buildFailureTable(kmpPattern2, comparator);
        int[] expected = {0, 0, 1, 0};
        assertArrayEquals(expected, failureTable2);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 4.", 4, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMPMatch2() {
        /*
            pattern: nano
            text: banananobano
            indices: 4
            expected total comparison: 24

            failure table: [0, 0, 1, 0]
            comparisons: 4

        b | a | n | a | n | a | n | o | b | a | n | o | 
        --+---+---+---+---+---+---+---+---+---+---+---+
        n | a | n | o |   |   |   |   |   |   |   |   | 
        - |   |   |   |   |   |   |   |   |   |   |   |  comparisons : 1
          | n | a | n | o |   |   |   |   |   |   |   |  
          | - |   |   |   |   |   |   |   |   |   |   |  comparisons : 1
          |   | n | a | n | o |   |   |   |   |   |   |  
          |   | - | - | - | - |   |   |   |   |   |   |  comparisons : 4
          |   |   |   | n | a | n | o |   |   |   |   | 
          |   |   |   |   | - | - | - |   |   |   |   |  comparisons : 3
          |   |   |   |   |   |   |   | n | a | n | o | 
          |   |   |   |   |   |   |   | - |   |   |   |  comparisons : 1



         comparisons: 10
         */
        List<Integer> kmpAnswer2 = new LinkedList<>();
        kmpAnswer2.add(4);
        assertEquals(kmpAnswer2,
                PatternMatching.kmp(kmpPattern2, kmpText2, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 14.", 14, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testAllMatchKMP() {
        /*
            pattern: sos
            text: sosososososososos
            indices: 0, 2, 4, 6, ... 14
            expected total comparison: 31
         */
        List<Integer> f = new ArrayList<>();
        f.add(0);
        f.add(2);
        f.add(4);
        f.add(6);
        f.add(8);
        f.add(10);
        f.add(12);
        f.add(14);
        assertEquals(f,
                PatternMatching.kmp("sos", "sosososososososos", comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 19.", 19, comparator.getComparisonCount());
    }



    @Test(timeout = TIMEOUT)
    public void testbuildFailureTable3() {
        int[] failureTable = PatternMatching
                .buildFailureTable("hhehhellolloo", comparator);
        int[] expected = {0, 1, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 14.", 14, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testbuildFailureTable4() {
        int[] failureTable = PatternMatching
                .buildFailureTable("", comparator);
        assertArrayEquals(new int[0], failureTable);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 0.", 0, comparator.getComparisonCount());
    }


    @Test(timeout = TIMEOUT)
    public void testKMPNoMatch() {
        /*
            pattern: 12345
            text: 123514231425342
            indices: -
            expected total comparison: 18

            failure table: [0, 0, 0, 0, 0]
            comparisons: 4

         */

        assertEquals(emptyList,
                PatternMatching.kmp(noMatchPattern, noMatchText , comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 10.", 18, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testLongerTextCheckKMP() {
        /*
            pattern: wherwherwherrhwherw
            text: where
            indices: -
            expected total comparison: 0
         */

        String longerText = "where";
        String longerPattern = "wherwherwherrhwherw";
        assertEquals(emptyList,
                PatternMatching.kmp(longerPattern, longerText, comparator));
        assertEquals(0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testEqualsKMP() {
        /*
            pattern: match!
            text: match!
            indices: 0
            expected total comparison: 11 (with failure table) or 5 (without FT)
         */

        String equalsText = "match!";
        String equalsPattern = "match!";
        assertEquals(kmpPatternEquivalencyAnswer,
                PatternMatching.kmp(equalsPattern, equalsText, comparator));
        assertTrue("Comparison count is different than expected",
                comparator.getComparisonCount() == 11
                        || comparator.getComparisonCount() == 5);
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable1() {
        /*
            pattern: banananobano
            last table: {s : 0, e : 1, l : 3}
         */

        String lt1 = "banananobano";
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable(lt1);
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        expectedLastTable.put('b', 8);
        expectedLastTable.put('a', 9);
        expectedLastTable.put('n', 10);
        expectedLastTable.put('o', 11);
        assertEquals(expectedLastTable, lastTable);
    }


    @Test(timeout = TIMEOUT)
    public void testBuildLastTable2() {
        /*
            pattern: _!_!__!!_
            last table: {_ : 8, ! : 7}
         */
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable("_!_!__!!_");
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        expectedLastTable.put('_', 8);
        expectedLastTable.put('!', 7);
        assertEquals(expectedLastTable, lastTable);
    }


    @Test(timeout = TIMEOUT)
    public void testBuildLastTable3() {
        /*
            pattern: ""
            last table: {}
         */
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable("");
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        assertEquals(expectedLastTable, lastTable);
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerExceptions() {
        assertThrows(IllegalArgumentException.class, () -> {
                PatternMatching.boyerMoore("null", "null", null);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.boyerMoore("", "null", comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.boyerMoore("null", null, comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
                PatternMatching.boyerMoore(null, "null", comparator);
        });
    }
    

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreMatch() {
        /*
            pattern: sell
            text: She sells seashells by the seashore.
            indices: 4
            expected total comparisons: 20
         */
        assertEquals(sellAnswer,
                PatternMatching.boyerMoore(sellPattern, sellText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 20.", 20, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreNoMatch() {
        /*
            pattern: 2023
            text: 202220213023021
            indices: -
            expected total comparisons: 11
         */
        assertEquals(emptyList,
                PatternMatching.boyerMoore("2023",
                        "202220213023021", comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 10.", 10, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreMultiMatch1() {
        /*
            pattern: sos
            text: bososmososmsdsossosossos
            indices: 0, 2
            expected total comparisons: 34
         */

        multipleAnswer = new ArrayList<>();
        multipleAnswer.add(2);
        multipleAnswer.add(7);
        multipleAnswer.add(13);
        multipleAnswer.add(16);
        multipleAnswer.add(18);
        multipleAnswer.add(21);

        assertEquals(multipleAnswer,
                PatternMatching.boyerMoore("sos",
                        "bososmososmsdsossosossos", comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 34.", 34, comparator.getComparisonCount());
    }
 

    @Test(timeout = TIMEOUT)
    public void boyerMooreAllMatch() {
        /*
            pattern: sos
            text: sosososososososos
            indices: 0, 2, 4, 6, ... 14
            expected total comparison: 31
         */
        List<Integer> finalAns = new ArrayList<>();
        finalAns.add(0);
        finalAns.add(2);
        finalAns.add(4);
        finalAns.add(6);
        finalAns.add(8);
        finalAns.add(10);
        finalAns.add(12);
        finalAns.add(14);
        assertEquals(finalAns,
                PatternMatching.boyerMoore("sos", "sosososososososos", comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 31.", 31, comparator.getComparisonCount());
    }


    @Test(timeout = TIMEOUT)
    public void testBoyerMooreLongerText() {
        /*
            pattern: waylonger
            text: w
            indices: -
            expected total comparisons: 0
         */
        assertEquals(emptyList,
                PatternMatching.boyerMoore("waylonger",
                        "w", comparator));
        assertEquals(0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpMatch1() {
        /*
            pattern: aaba
            text: aabaacaadaabaaba
            indices: 0, 9, 12
            expected total comparison: 12
         */
        assertEquals(kmpAnswer,
                PatternMatching.rabinKarp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 12.", 12, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpNoMatch() {
        /*
            pattern: abcde
            text: abceadbcadbecdb
            indices: -
            expected total comparison: 0
         */

        assertEquals(emptyList,
        PatternMatching.rabinKarp("abcde", "abceadbcadbecdb", comparator));
        assertEquals("Comparison count was " + comparator.getComparisonCount()
         + ". Should be 0.", 0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpMatch2() {
        /*
            pattern: nano
            text: banananobano
            indices: 4
            expected total comparison: 24

         */
        List<Integer> rabin2 = new LinkedList<>();
        rabin2.add(4);
        assertEquals(rabin2,
                PatternMatching.rabinKarp(kmpPattern2, kmpText2, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 4.", 4, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpLongerText() {
        /*
            pattern: wherwherwherrhwherw
            text: where
            indices: -
            expected total comparison: 0
        */

        String longerText = "where";
        String longerPattern = "wherwherwherrhwherw";
        assertEquals(emptyList,
                 PatternMatching.rabinKarp(longerPattern, longerText, comparator));
        assertEquals(0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpAllMatch() {
        /*
            pattern: sos
            text: sosososososososos
            indices: 0, 2, 4, 6, ... 14
            expected total comparison: 31
         */
        List<Integer> f = new ArrayList<>();
        f.add(0);
        f.add(2);
        f.add(4);
        f.add(6);
        f.add(8);
        f.add(10);
        f.add(12);
        f.add(14);
        assertEquals(f,
                PatternMatching.rabinKarp("sos", "sosososososososos", comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 24.", 24, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpEqualHashes() {
        /*
            These are characters with ASCII values as shown, not the actual
            characters shown. Most do not have actual characters.
        

            pattern: 011
            text: 00101011(114)
            indices: 5
            expected total comparisons: 5
         */
        List<Integer> answer = new ArrayList<>();
        answer.add(5);
        assertEquals(answer,
                PatternMatching.rabinKarp("\u0000\u0001\u0001",
                        "\u0000\u0000\u0001\u0000\u0001\u0000\u0001\u0001\u0072", comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 3.", 3, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpMultiMatch() {
        /*
            pattern: sos
            text: bososmososmsdsossosossos
            indices: 0, 2
            expected total comparisons: 34
         */

        multipleAnswer = new ArrayList<>();
        multipleAnswer.add(2);
        multipleAnswer.add(7);
        multipleAnswer.add(13);
        multipleAnswer.add(16);
        multipleAnswer.add(18);
        multipleAnswer.add(21);

        assertEquals(multipleAnswer,
                PatternMatching.rabinKarp("sos",
                        "bososmososmsdsossosossos", comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 18.", 18, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpExceptions() {
        assertThrows(IllegalArgumentException.class, () -> {
                PatternMatching.rabinKarp("null", "null", null);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.rabinKarp("", "null", comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.rabinKarp("null", null, comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
                PatternMatching.rabinKarp(null, "null", comparator);
        });
    }


}
