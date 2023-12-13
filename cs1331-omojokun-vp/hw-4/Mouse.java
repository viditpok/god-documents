/**
*This class represents a Mouse object.
*@author Vidit Pokharna
*@version 1.0
*/
public class Mouse {
    /**
     *
     */
    private double mass;
    /**
     *
     */
    private double speed;

    /**
         * @param mass1 mass input
         * @param speed1 speed input
         */
        public Mouse(double mass1, double speed1) {
        this.mass = mass1;
        this.speed = speed1;
    }

    /**
     * @param mass2 mass input
     */
    public Mouse(double mass2) {
        this(mass2, 10);
    }

    /**
         *
         */
        public Mouse() {
        this(50, 10);
    }

    /**
    *
    */
    public void consumeCheese() {
        mass += 20;
        if (mass < 100) {
            speed += 1;
        } else {
            speed -= 0.5;
        }
    }

        /**
         * @return boolean whether mouse is dead or not
         */
    public boolean isDead() {
        boolean h = false;
        if (mass == 0) {
            return true;
        }
        return h;
    }


    /**
     * @return String
     */
    public String toString() {
        String mass3 = String.format("%.2f", mass);
        String speed3 = String.format("%.2f", speed);
        if (mass == 0) {
            return ("I'm dead, but I used to be a mouse with a speed of " + speed3 + ".");
        } else {
            return ("I'm a speedy mouse with " + speed3 + " speed and " + mass3 + " mass.");
        }
    }


    /**
     * @return double
     */
    public double getMass() {
        return mass;
    }


    /**
     * @param setM mass input
     */
    public void setMass(double setM) {
        this.mass = setM;
    }


    /**
     * @return double
     */
    public double getSpeed() {
        return speed;
    }


    /**
     * @param setS speed input
     */
    public void setSpeed(double setS) {
        this.speed = setS;
    }
}