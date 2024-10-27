package ch.weetech.vector;

import jdk.incubator.vector.*;
import java.util.random.RandomGenerator;

/*
 * https://inside.java/2024/10/23/java-and-ai/
 */
public class VectorizedArrayAddition {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void main(String[] args) {
        // Define the size of the arrays
        // ensuring that it's a multiple of SPECIES.length() for simplicity
        int size = 4 * SPECIES.length() ;

        // Create two arrays
        float[] arrayA = new float[size];
        float[] arrayB = new float[size];

        // Generate random values for arrayA and arrayB
        RandomGenerator random = RandomGenerator.getDefault();
        for (int i = 0; i < size; i++) {
            arrayA[i] = random.nextFloat();
            arrayB[i] = random.nextFloat();
        }

        // Result array to store the sum
        float[] result = new float[size];

        // Perform element-wise addition using the Vector API
        int i = 0;
        for (; i < SPECIES.loopBound(size); i += SPECIES.length()) {
            // Load elements from the arrays into vectors
            FloatVector va = FloatVector.fromArray(SPECIES, arrayA, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, arrayB, i);

            // Perform vectorized addition
            FloatVector resultVector = va.add(vb);

            // Store the result back into the result array
            resultVector.intoArray(result, i);
        }

        // Print the arrays and the result
        System.out.println("Array A: ");
        printArray(arrayA);
        System.out.println("Array B: ");
        printArray(arrayB);
        System.out.println("Result (A + B): ");
        printArray(result);
    }

    // Helper method to print an array
    private static void printArray(float[] array) {
        for (float value : array) {
            System.out.printf("%.4f ", value);
        }
        System.out.println();
    }

}
