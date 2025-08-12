package yoshikihigo.tinypdg.graphviz;

import org.apache.commons.cli.*;
import org.apache.commons.io.FileUtils;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.json.JSONObject;
import yoshikihigo.tinypdg.ast.TinyPDGASTVisitor;
import yoshikihigo.tinypdg.cfg.CFG;
import yoshikihigo.tinypdg.cfg.edge.CFGEdge;
import yoshikihigo.tinypdg.cfg.node.CFGControlNode;
import yoshikihigo.tinypdg.cfg.node.CFGNode;
import yoshikihigo.tinypdg.cfg.node.CFGNodeFactory;
import yoshikihigo.tinypdg.pdg.PDG;
import yoshikihigo.tinypdg.pdg.edge.PDGControlDependenceEdge;
import yoshikihigo.tinypdg.pdg.edge.PDGDataDependenceEdge;
import yoshikihigo.tinypdg.pdg.edge.PDGEdge;
import yoshikihigo.tinypdg.pdg.edge.PDGExecutionDependenceEdge;
import yoshikihigo.tinypdg.pdg.node.*;
import yoshikihigo.tinypdg.pe.MethodInfo;
import yoshikihigo.tinypdg.pe.ProgramElementInfo;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class WriterPdgToJsonTest {
    public static String jsonPath = "";
    public static void main(String[] args) {

        try {

            final Options options = new Options();

            {
                final Option d = new Option("d", "directory", true,
                        "target directory");
                d.setArgName("directory");
                d.setArgs(1);
                d.setRequired(true);
                options.addOption(d);
            }

            {
                final Option c = new Option("c", "ControlFlowGraph", true,
                        "control flow graph");
                c.setArgName("file");
                c.setArgs(1);
                c.setRequired(false);
                options.addOption(c);
            }

            {
                final Option p = new Option("p", "ProgramDepencencyGraph",
                        true, "program dependency graph");
                p.setArgName("file");
                p.setArgs(1);
                p.setRequired(false);
                options.addOption(p);
            }

            final CommandLineParser parser = new PosixParser();
            final CommandLine cmd = parser.parse(options, args);

            final File target = new File(cmd.getOptionValue("d"));
            jsonPath = cmd.getOptionValue("d");
            if (!target.exists()) {
                System.err
                        .println("specified directory or file does not exist.");
                System.exit(0);
            }

            final List<File> files = getFiles(target);
            final List<MethodInfo> methods = new ArrayList<MethodInfo>();
            for (final File file : files) {
                final CompilationUnit unit = TinyPDGASTVisitor.createAST(file);
                final List<MethodInfo> m = new ArrayList<MethodInfo>();
                final TinyPDGASTVisitor visitor = new TinyPDGASTVisitor(
                        file.getAbsolutePath(), unit, methods);
                unit.accept(visitor);
                methods.addAll(m);
            }

            if (cmd.hasOption("p")) {
                //System.out.println("building and outputing PDGs ...");
                final BufferedWriter writer = new BufferedWriter(
                        new FileWriter(cmd.getOptionValue("p")));

                writer.write("digraph {");
                writer.newLine();

                int createdGraphNumber = 0;
                //拼接多个方法生成的json，要在头加“[”，尾部判断加“,”或者“]”
                boolean startFlag;
                boolean endFlag;
                for(int i = 0;i < methods.size();i++){
                    MethodInfo method = methods.get(i);
                    final PDG pdg = new PDG(method, new PDGNodeFactory(),
                            new CFGNodeFactory(), true, true, true);
                    pdg.build();
                    startFlag = i == 0;
                    endFlag = i == methods.size() - 1;
                    writePDG(method.getName(),pdg, createdGraphNumber++, writer,startFlag, endFlag);
                }

                writer.write("}");

                writer.close();
            }

            //System.out.println("successfully finished.");

        } catch (Exception e) {
            System.err.println("Exception"+e.getMessage());
            System.exit(0);
        }
    }

    static private void writePDG(final String methodName,final PDG pdg, final int createdGraphNumber,
                                 final BufferedWriter writer,boolean startFlag,boolean endFlag) throws IOException {

        final MethodInfo method = pdg.unit;

        writer.write("subgraph cluster");
        writer.write(Integer.toString(createdGraphNumber));
        writer.write(" {");
        writer.newLine();

        if (createdGraphNumber>0){
            System.out.println("createdGraphNumber "+createdGraphNumber);
        }

        writer.write("label = \"");
        writer.write(getMethodSignature(method));
        String functionName = getMethodSignature(method).split(" ")[0];

        writer.write("\";");
        writer.newLine();

        final Map<PDGNode<?>, Integer> nodeLabels = new HashMap<PDGNode<?>, Integer>();
        //System.out.println("pdg.getAllNodes()"+(pdg.getAllNodes()).size());/////////////////////////////////////////////////////////////
        for (final PDGNode<?> node : pdg.getAllNodes()) {
            nodeLabels.put(node, nodeLabels.size());
        }

        //Create a json format data of PDG
        JSONObject methodCodeGraphJson = new JSONObject();
        JSONObject codeGraphJson = new JSONObject();
        JSONObject jsonNodes = new JSONObject();
        JSONObject jsonEdges = new JSONObject();

        JSONObject methodCodeGraphJsonVec = new JSONObject();
        JSONObject codeGraphJsonVec = new JSONObject();
        JSONObject jsonNodesVec = new JSONObject();
        JSONObject jsonEdgesVec = new JSONObject();


        BufferedWriter pdg_corpus = new BufferedWriter(new FileWriter("./outPut_pdg/corpus/pdg_corpus.txt",true));
        for (final Map.Entry<PDGNode<?>, Integer> entry : nodeLabels.entrySet()) {    /*####################################################### Nodes*/
            writer.write(Integer.toString(createdGraphNumber));
            writer.write(".");
            writer.write(Integer.toString(entry.getValue()));
            writer.write(" [style = filled, label = \"");
            //DATASET-去除多余空格-添加括号分号等符号的版本
            String nodeString = entry.getKey().getText().replace("\"", "\\\"")
                    .replace("\\\\\"", "\\\\\\\"").replace(";"," ; ")
                    .replace("\n", " \\n ").replace("\r", " \\r ")
                    .replace("(", " ( ").replace(")", " ) ")
                    .replace("[", " [ ").replace("]", " ] ")
                    .replace(" (  ) ", " () ").replace(" [  ] ", " [] ")
                    .replace("[][]", " [][] ").replace("="," = ")
                    .replace(".", " . ").replace(":"," : ").replace(","," , ")
                    .replace("+"," + ").replace("-"," - ").replace("+  +","++")
                    .replace("-  -","--").replace("~"," ~ ").replace("!"," ! ")
                    .replace("*"," * ").replace("/"," / ").replace("%"," % ")
                    .replace("<"," < ").replace(">"," > ").replace(">  >",">>")
                    .replace(">  >  >",">>>")
                    .replace("<  <","<<").replace("<  =","<=")
                    .replace(">  =",">=").replace("instanceof"," instanceof ").replace("=  =","==")
                    .replace("!  =","!=").replace("&"," & ").replace("|"," | ")
                    .replace("^"," ^ ").replace("&  &","&&").replace("|  |","||")
                    .replace("? :"," ?:").replace("+  =","+=")
                    .replace("-  =","-=").replace("*  =","*=").replace("/  =","/=")
                    .replace("%  =","%=").replace("&  =","&=").replace("|  =","|=").replace("/  /","//")
                    .replace("^  =","^=").replace("<  <  =","<<=").replace(">  >   =",">>=")
                    .replace(">  >  >  =",">>>=").replaceAll("\\s+"," ");

            /*DATASET-去除多余空格-去除括号分号逗号点等版本
            String nodeString = entry.getKey().getText().replace("\"", "\\\"")
                    .replace("\\\\\"", "\\\\\\\"").replace(";"," ")
                    .replace("\n", " \\n ").replace("\r", " \\r ")
                    .replace("(", " ").replace(")", " ")
                    .replace("[", " ").replace("]", " ")
                    .replace(" (  ) ", " () ").replace(" [  ] ", " [] ")
                    .replace("[][]", " [][] ").replace("="," = ")
                    .replace(".", " ").replace(":"," : ").replace(","," ")
                    .replace("+"," + ").replace("-"," - ").replace("+  +","++")
                    .replace("-  -","--").replace("~"," ~ ").replace("!"," ! ")
                    .replace("*"," * ").replace("/"," / ").replace("%"," % ")
                    .replace("<"," < ").replace(">"," > ").replace(">  >",">>")
                    .replace(">  >  >",">>>")
                    .replace("<  <","<<").replace("<  =","<=")
                    .replace(">  =",">=").replace("instanceof"," instanceof ").replace("=  =","==")
                    .replace("!  =","!=").replace("&"," & ").replace("|"," | ")
                    .replace("^"," ^ ").replace("&  &","&&").replace("|  |","||")
                    .replace("? :"," ?:").replace("+  =","+=")
                    .replace("-  =","-=").replace("*  =","*=").replace("/  =","/=")
                    .replace("%  =","%=").replace("&  =","&=").replace("|  =","|=").replace("/  /","//")
                    .replace("^  =","^=").replace("<  <  =","<<=").replace(">  >   =",">>=")
                    .replace(">  >  >  =",">>>=").replaceAll("\\s+"," ");
             */
            if(nodeString.equals("Enter")){
                nodeString = "Enter " + functionName + " ";
            }
            writer.write(nodeString);
            writer.write("\"");

            final PDGNode<?> node = entry.getKey();
            // 写入文件
            pdg_corpus.write(nodeString);
            String[] code = nodeString.split("\\s+");
            float[][] vectors = new float[code.length][createPDGCodeJson.vec.getWordVector("int").length];
            for(int i=0;i< code.length;i++){
                try{
                    vectors[i] = createPDGCodeJson.vec.getWordVector(code[i]);
                    if(vectors[i] == null){
                        vectors[i] = new float[16];
                    }
                } catch (Exception e){
                    e.printStackTrace();
                    System.out.println("vectvectors[i]vectors[i]vectors[i]vectors[i]vectors[i]ors[i]");
                }
            }

            jsonNodes.put(Integer.toString(entry.getValue()),nodeString);
            jsonNodesVec.put(Integer.toString(entry.getValue()),vectors);
            //System.out.println("jsonNodes "+jsonNodes);



            if (node instanceof PDGMethodEnterNode) {
                writer.write(", fillcolor = aquamarine");
            } else if (pdg.getExitNodes().contains(node)) {
                writer.write(", fillcolor = deeppink");
            } else if (node instanceof PDGParameterNode) {
                writer.write(", fillcolor = tomato");
            } else {
                writer.write(", fillcolor = white");
            }

            if (node instanceof PDGControlNode) {
                writer.write(", shape = diamond");
            } else if (node instanceof PDGParameterNode) {
                writer.write(", shape = box");
            } else {
                writer.write(", shape = ellipse");
            }

            writer.write("];");
            writer.newLine();

        }
        pdg_corpus.newLine();
        pdg_corpus.close();
        //System.out.println("pdg.getAllEdges()"+(pdg.getAllEdges()).size());/////////////////////////////////////////////////////////////
        for (final PDGEdge edge : pdg.getAllEdges()) {                        /*####################################################### Edges*/
            writer.write(Integer.toString(createdGraphNumber));
            writer.write(".");
            writer.write(Integer.toString(nodeLabels.get(edge.fromNode)));
            writer.write(" -> ");
            writer.write(Integer.toString(createdGraphNumber));
            writer.write(".");
            writer.write(Integer.toString(nodeLabels.get(edge.toNode)));

            // 写入文件
            jsonEdges.put(nodeLabels.get(edge.fromNode) +"->"+ nodeLabels.get(edge.toNode),edge.getDependenceString());


            String[] code = edge.getDependenceString().replace(";"," ;").replace("\n", "\\n").replace("\r", "\\r").split("\\s+");
            float[][] vectors = new float[code.length][createPDGCodeJson.vec.getWordVector("int").length];
            if(code[0].equals("")){
                Arrays.fill(vectors[0], 1);
            }else{
                for(int i=0;i< code.length;i++){
                    vectors[i] = createPDGCodeJson.vec.getWordVector(code[i]);
                    if(vectors[i] == null){
                        vectors[i] = new float[16];
                    }
                }
            }

            jsonEdgesVec.put(nodeLabels.get(edge.fromNode) +"->"+ nodeLabels.get(edge.toNode),vectors);


            if (edge instanceof PDGDataDependenceEdge) {
                writer.write(" [style = solid, label=\""
                        + edge.getDependenceString() + "\"]");
            } else if (edge instanceof PDGControlDependenceEdge) {
                writer.write(" [style = dotted, label=\""
                        + edge.getDependenceString() + "\"]");
            } else if (edge instanceof PDGExecutionDependenceEdge) {
                writer.write(" [style = bold, label=\""
                        + edge.getDependenceString() + "\"]");
            }
            writer.write(";");
            writer.newLine();
        }

        writer.write("}");
        writer.newLine();

        //System.out.println(codeGraphJson);
        codeGraphJson.put("jsonNodes",jsonNodes);
        codeGraphJson.put("jsonEdges",jsonEdges);
        methodCodeGraphJson.put(methodName,codeGraphJson);

        codeGraphJsonVec.put("jsonNodesVec",jsonNodesVec);
        codeGraphJsonVec.put("jsonEdgesVec",jsonEdgesVec);
        methodCodeGraphJsonVec.put(methodName,codeGraphJsonVec);

        writeToFile(startFlag, endFlag, methodCodeGraphJson, methodCodeGraphJsonVec);
    }

    private static void writeToFile(boolean startFlag, boolean endFlag, JSONObject codeGraphJson, JSONObject codeGraphJsonVec) {
        try {
            //System.out.println("jsonPath PDG"+jsonPath);
            String[] pathParts = jsonPath.split(File.separator.equals("\\") ? "\\\\" : "/");
            StringBuilder codeJsonPath = new StringBuilder("./outPut_pdg/codeJson/");
            StringBuilder codeJsonVecPath = new StringBuilder("./outPut_pdg/codeJsonVec/");
            for(int j = 4; j < pathParts.length - 1 ;j++){
                codeJsonPath.append(pathParts[j]).append(File.separator);
                codeJsonVecPath.append(pathParts[j]).append(File.separator);
            }
            File codeJsonFile = new File(codeJsonPath.toString());
            if(!codeJsonFile.exists()){//如果文件夹不存在
                codeJsonFile.mkdirs();//创建文件夹
            }
            String start;
            String end;
            //通过判断，把一个java文件中的多个method的json拼接起来
            if(startFlag){
                start = "[";
            }else {
                start = "";
            }
            if(endFlag){
                end = "]";
            }else {
                end = ",";
            }
            FileUtils.write(new File(codeJsonPath+ File.separator + pathParts[pathParts.length - 1]+".json"),
                    start + codeGraphJson.toString() + end, StandardCharsets.UTF_8, true);

            File codeJsonVecFile = new File(codeJsonVecPath.toString());
            if(!codeJsonVecFile.exists()){//如果文件夹不存在
                codeJsonVecFile.mkdirs();//创建文件夹
            }
            FileUtils.write(new File(codeJsonVecPath+ File.separator + pathParts[pathParts.length - 1]+".json"),
                    start + codeGraphJsonVec.toString() + end, StandardCharsets.UTF_8, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static private List<File> getFiles(final File file) {

        final List<File> files = new ArrayList<File>();

        if (file.isFile() && file.getName().endsWith(".java")) {
            files.add(file);
        }

        else if (file.isDirectory()) {
            for (final File child : file.listFiles()) {
                final List<File> children = getFiles(child);
                files.addAll(children);
            }
        }

        return files;
    }

    static private String getMethodSignature(final MethodInfo method) {

        final StringBuilder text = new StringBuilder();

        text.append(method.name);
        text.append(" <");
        text.append(method.startLine);
        text.append("...");
        text.append(method.endLine);
        text.append(">");

        return text.toString();
    }
}
