/**
*This class represents a Cat object.
*@author Vidit Pokharna
*@version 1.0
*/
public class Cat {
    /**
     *
     */
    private static String breed = "Persian Tabby";
    /**
     *
     */
    private String name;
    /**
     *
     */
    private int age;
    /**
     *
     */
    private double runningSpeed;
    /**
     *
     */
    private boolean isKitten;

    /**
     * @param name1 name input
     * @param age1 age input
     * @param runningSpeed1 speed input
     */
    public Cat(String name1, int age1, double runningSpeed1) {
        this.name = name1;
        this.age = age1;
        this.runningSpeed = runningSpeed1;
        if (age > 6) {
            this.isKitten = false;
        } else {
            this.isKitten = true;
        }
    }
    /**
     * @param name2 name input
     * @param ageInYears age input
     * @param runningSpeed2 speed input
     */
    public Cat(String name2, double ageInYears, double runningSpeed2) {
        this(name2, (int) (ageInYears * 12), runningSpeed2);
    }
    /**
     * @param name3 name input
     */
    public Cat(String name3) {
        this(name3, 5, 5);
    }
    /**
     * @param months month input
     */
    public void increaseAge(int months) {
        for (int a = 0; a < months; a++) {
            age++;
            if (age <= 12) {
                runningSpeed += 2;
            } else if (age > 30) {
                if (runningSpeed > 8) {
                    runningSpeed -= 3;
                }
            }
            if (age > 6) {
                isKitten = false;
            } else {
                isKitten = true;
            }
        }
    }
    /**
     *
     */
    public void increaseAge() {
        if (age <= 12) {
            runningSpeed += 2;
            age++;
        } else if (age > 30) {
            if (runningSpeed > 8) {
                runningSpeed -= 3;
                age++;
            }
        }
        if (age < 6) {
            isKitten = true;
        } else {
            isKitten = false;
        }
    }
    /**
     * @param m mouse input
     */
    public void eat(Mouse m) {
        if (m.isDead()) {
            return;
        } else {
            if (runningSpeed > m.getSpeed()) {
                if (m.getMass() >= (0.65 * age)) {
                    increaseAge();
                }
                m.setMass(0);
            } else {
                m.consumeCheese();
            }
        }
    }

    /**
     * @return String
     */
    public String toString() {
        String speed4 = String.format("%.2f", runningSpeed);
        if (isKitten) {
            String ret = ("My name is " + name  + " and I'm a Kitten! I'm " + age + " months");
            ret += (" old and I run at the speed of " + speed4 + ".");
            return ret;
        } else {
            String ret1 = ("My name is " + name + " and I'm a " + breed + ". I'm " + age);
            ret1 += (" months old and I run at the speed of " + speed4 + ".");
            return ret1;
        }
    }
    /**
     * @return String
     */
    public static String getBreed() {
        return breed;
    }
    /**
     * @param setB breed input
     */
    public static void setBreed(String setB) {
        breed = setB;
    }
    /**
     * @return name
     */
    public String getName() {
        return name;
    }
    /**
     * @param name name input
     */
    public void setName(String name) {
        this.name = name;
    }
    /**
     * @return age
     */
    public int getAge() {
        return age;
    }
    /**
     * @param age age input
     */
    public void setAge(int age) {
        this.age = age;
        if (age > 6) {
            this.isKitten = false;
        } else {
            this.isKitten = true;
        }
    }
    /**
     * @return speed
     */
    public double getRunningSpeed() {
        return runningSpeed;
    }
    /**
     * @param speed speed input
     */
    public void setRunningSpeed(double speed) {
        this.runningSpeed = speed;
    }
    /**
     * @return boolean stating whether cat is kitten
     */
    public boolean getIsKitten() {
        return isKitten;
    }
}