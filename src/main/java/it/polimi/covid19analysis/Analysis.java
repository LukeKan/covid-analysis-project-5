package it.polimi.covid19analysis;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeoutException;

import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.date_sub;

public class Analysis {
    public static void main(String[] args) {
        final String master = args.length > 0 ? args[0] : "local[4]";
        final String socketHost = args.length > 1 ? args[1] : "localhost";
        final int socketPort = args.length > 2 ? Integer.parseInt(args[2]) : 9999;
        final String filePath = args.length > 3 ? args[3] : "./files/covid-data/";
        final String fileName = args.length > 4 ? args[4] : "covid_data.csv";

        long startTime = System.nanoTime();

        SparkSession spark = SparkSession
                .builder()
                .appName("COVID analysis")
                .master(master)
                .getOrCreate();

        final List<StructField> covidDataFields = new ArrayList<>();
        covidDataFields.add(DataTypes.createStructField
                ("dateRep", DataTypes.DateType, false));
        covidDataFields.add(DataTypes.createStructField
                ("day", DataTypes.IntegerType, false));
        covidDataFields.add(DataTypes.createStructField
                ("month", DataTypes.IntegerType, false));
        covidDataFields.add(DataTypes.createStructField
                ("year", DataTypes.IntegerType, false));
        covidDataFields.add(DataTypes.createStructField
                ("cases", DataTypes.IntegerType, false));
        covidDataFields.add(DataTypes.createStructField
                ("countriesAndTerritories", DataTypes.StringType, false));
        StructType schema = DataTypes
                .createStructType(covidDataFields);

        final Dataset<Row> dataset = spark.read()
                .option("header", "false")
                .option("delimiter", ";")
                .schema(schema)
                .csv(filePath+fileName);



        final Dataset<Row> movingAverage = dataset.as("d1")
                .join(dataset.as("d2"),
                        col("d2.countriesAndTerritories").equalTo(col("d1.countriesAndTerritories"))
                                .and(col("d2.dateRep").leq(
                                        col("d1.dateRep")))
                                .and(col("d2.dateRep").geq(
                                        date_sub(col("d1.dateRep"),6).cast(DataTypes.DateType)))
                )
                .groupBy(col("d1.dateRep"),col("d1.countriesAndTerritories"))
                .agg(count("d2.dateRep").as("count"),avg(col("d2.cases")).alias("weekly_cases"))

                .filter(col("count").equalTo(7))
                .sort(col("d1.countriesAndTerritories"),col("d1.dateRep"))
                .select("dateRep","countriesAndTerritories","weekly_cases");


        movingAverage.show(10000);

        final Dataset<Row> percentageIncreases = movingAverage.as("m1")
                .join(movingAverage.as("m2"),
                        col("m1.countriesAndTerritories").equalTo(col("m2.countriesAndTerritories"))
                                .and(col("m1.dateRep").equalTo(date_add(col("m2.dateRep"),1))))
                .filter(col("m2.weekly_cases").notEqual(0)) // this removes infinite percentage increases
                .withColumn("daily_increase",
                        col("m1.weekly_cases").minus(col("m2.weekly_cases")))
                .withColumn("daily_percentage",
                        col("daily_increase").divide(col("m2.weekly_cases")).multiply(100))
                .sort(desc("m1.dateRep"),desc("daily_percentage"))
                .select("m1.dateRep","m1.countriesAndTerritories","daily_percentage");

        percentageIncreases.show(100);


        final Dataset<Row> top10PerDay = percentageIncreases.as("p1")
                .withColumn("rank", row_number().over(
                        Window.partitionBy("p1.dateRep")
                        .orderBy(desc("p1.daily_percentage")))
                )
                .where(col("rank").leq(10))
                .select(col("p1.dateRep"),  col("p1.countriesAndTerritories"), col("p1.daily_percentage"))
                .sort(desc("p1.dateRep"),desc("p1.daily_percentage"));
        top10PerDay.show(100);

        long stopTime = System.nanoTime();
        System.out.println((stopTime - startTime) / 1e9);
        spark.stop();
        //benchmark();

        /*final Dataset<Row> countCountry = dataset
                .groupBy("dateRep")
                .agg(count("countriesAndTerritories").name("count"))
                .agg(max("count"));

        countCountry.show(10);
        spark.stop();*/
    }

}
