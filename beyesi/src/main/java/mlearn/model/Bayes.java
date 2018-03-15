package mlearn.model;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created with IntelliJ IDEA.
 * User: hui
 * Date: 2018/3/13 21:03
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:   bayes 主体类
 */
public class Bayes {

    protected static Log LOG = LogFactory.getLog(Bayes.class);

    /**
     * 模型预测
     *
     * @param datas       训练集
     * @param testVectors 测试集向量
     * @return
     */
    public static String modelPredict(ArrayList<ArrayList<String>> datas, ArrayList<String> testVectors) {

        // 如果训练集和测试集都为空，则返回空
        if (datas == null || testVectors == null) {
            return null;
        }
        // 训练集按标签分好类的结果
        Map<String, ArrayList<ArrayList<String>>> map = classByLabel(datas);
        // 训练集的样本标签
        Object[] labels = map.keySet().toArray();
        // 预测标签的下标
        int preLabelIndex = -1;
        // 最大的预测值
        double maxPreRes = 0.0;
        // 对训练集相同的样本标签进行遍历
        for (int i = 0; i < map.size(); i++) {
            double tempRes = 0.0;
            // 对测试集进行遍历
            for (String testData : testVectors) {
                tempRes += modelPreExecutor(map, labels[i].toString(), testData);
            }
            if (tempRes > maxPreRes) {
                maxPreRes = tempRes;
                preLabelIndex = i;
            }
        }
        return preLabelIndex == -1 ? "其他" : labels[preLabelIndex].toString();
    }

    /**
     * 将训练集按标签划分
     *
     * @param datas 训练集
     * @return map(标签类型，对应的训练集)
     */
    public static Map<String, ArrayList<ArrayList<String>>> classByLabel(ArrayList<ArrayList<String>> datas) {
        // 如果数据为空则返回空
        if (datas == null) {
            return null;
        }

        HashMap<String, ArrayList<ArrayList<String>>> resMap = new HashMap<String, ArrayList<ArrayList<String>>>();
        ArrayList<String> dataLine = null;
        String label = null;
        // 逐行便利训练集
        for (ArrayList<String> data : datas) {
            // 获取一行数据
            dataLine = data;
            // 获取样本标签
            label = dataLine.get(0);
            // 删除样本标签
            dataLine.remove(0);
            // 如果包含，则将该数据放在对应的 map 中
            if (resMap.containsKey(label)) {
                resMap.get(label).add(dataLine);
            } else {
                // 初始化二维矩阵
                ArrayList<ArrayList<String>> lists = new ArrayList<ArrayList<String>>();
                // 添加一行数据
                lists.add(dataLine);
                resMap.put(label, lists);
            }
        }
        return resMap;
    }

    /**
     * 计算在出现 key 情况下，是分类 class 1 的概率 [ P(Classify | key) ]
     *
     * @param trainSet      所有分类后的数据集
     * @param classify 某一特定分类
     * @param key      某一特定特征
     * @return
     */
    public static double modelPreExecutor(Map<String, ArrayList<ArrayList<String>>> trainSet, String classify, String key) {
        ArrayList<ArrayList<String>> singles = trainSet.get(classify);
        double pkc = excutorKeyInClass(singles, key); // p(key|classify)
        double pc = excutorClass(trainSet, classify);    // p(classify)
        double pk = excutorKey(trainSet, key); // p(key)
        return pk == 0.0 ? 0.0 : pkc * pc / pk; // p(classify | key)
    }

    /**
     * 在某一特征值在某一分类上的概率分布 [ P(key|Classify) ]
     *
     * @param singles 某一分类上的训练集
     * @param key     测试集
     * @return
     */
    public static double excutorKeyInClass(ArrayList<ArrayList<String>> singles, String key) {

        if (singles == null || StringUtils.isBlank(key)) {
            return 0.0;
        }

        int totalKeyCOunt = 0;
        int foundKeyCount = 0;
        for (ArrayList<String> trainVector : singles) {
            for (String data : trainVector) {
                totalKeyCOunt++;
                if (data.equalsIgnoreCase(key)) {
                    foundKeyCount++;
                }
            }
        }
        return totalKeyCOunt == 0 ? 0.0 : 1.0 * foundKeyCount / totalKeyCOunt;
    }

    /**
     * 获得某一分类的概率 [ P(Classify) ]
     *
     * @param map      全部的训练集
     * @param classify 某一分类
     * @return
     */
    public static double excutorClass(Map<String, ArrayList<ArrayList<String>>> map, String classify) {
        if (map == null || StringUtils.isBlank(classify)) {
            return 0.0;
        }
        int totalClassifyCount = 0;
        Set<String> labelSet = map.keySet();
        Iterator<String> iterator = labelSet.iterator();
        while (iterator.hasNext()) {
            totalClassifyCount += map.get(iterator.next()).size();
        }
        return 1.0 * map.get(classify).size() / totalClassifyCount;
    }

    /**
     * 获得关键词的总概率 [ P(key) ]
     *
     * @param map 总的训练集
     * @param key 测试集
     * @return
     */
    public static double excutorKey(Map<String, ArrayList<ArrayList<String>>> map, String key) {
        if (map == null || StringUtils.isBlank(key)) {
            return 0.0;
        }

        int foundKeyCount = 0;
        int totalKeyCount = 0;

        Set<String> labelSet = map.keySet();
        for (String label : labelSet) {
            ArrayList<ArrayList<String>> singles = map.get(label);
            for (ArrayList<String> vector : singles) {
                for (String data : vector) {
                    totalKeyCount++;
                    if (data.equalsIgnoreCase(key)) {
                        foundKeyCount++;
                    }
                }
            }
        }
        return totalKeyCount == 0 ? 0.0 : 1.0 * foundKeyCount / totalKeyCount;
    }

    /**
     * 读取训练集数据
     *
     * @param trainPath   训练集所存放的路径
     * @return
     */
    public static ArrayList<ArrayList<String>> read(String trainPath) throws Exception{
        ArrayList<String> singleLabel = null;
        ArrayList<ArrayList<String>> trainSet = new ArrayList<ArrayList<String>>();
        try {
            List<String> lines = FileUtils.readLines(new File(trainPath), "UTF-8");
            if (lines.size() == 0){
                LOG.info("训练数据为空" + trainPath);
                throw new Exception("训练数据为空！");
            }
            for (String line : lines) {
                String[] split = line.split(" ");
                singleLabel = new ArrayList<String>();
                for (String s : split) {
                    if (StringUtils.isNotBlank(s)){
                        singleLabel.add(s);
                    }
                }
                trainSet.add(singleLabel);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return trainSet;
    }

    public static void main(String[] args) throws Exception{
        String path = "beyesi/src/main/resources/trainData.txt";
        ArrayList<ArrayList<String>> trainSet = Bayes.read(path);
        ArrayList<String> testData = new ArrayList<String>();
        testData.add("海贼王");
        testData.add("罗杰");
        String label = Bayes.modelPredict(trainSet, testData);
        System.out.println(label);
    }

}