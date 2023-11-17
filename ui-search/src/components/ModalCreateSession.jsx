import {
  Box,
  Checkbox,
  ColumnLayout,
  ExpandableSection,
  Form,
  FormField,
  Input,
  Select,
  Textarea,
  Tiles,
  Toggle,
} from '@cloudscape-design/components';
import Button from '@cloudscape-design/components/button';
import Modal from '@cloudscape-design/components/modal';
import SpaceBetween from '@cloudscape-design/components/space-between';
import { useCallback, useEffect, useState } from 'react';
import useIndexNameList from 'src/hooks/useIndexNameList';
import useInput from 'src/hooks/useInput';
import useLsLanguageModelList from 'src/hooks/useLsLanguageModelList';
import useToggle from 'src/hooks/useToggle';
import { useSessionStore } from 'src/stores/session';
import Divider from './Divider';
import useLsSessionList from 'src/hooks/useLsSessionList';
import useLsAppConfigs from 'src/hooks/useLsAppConfigs';

const SIZE = 's';
const OPTIONS_LANGUAGE = [
  { label: '简体中文', value: 'chinese' },
  { label: '繁体中文', value: 'chinese-tc' },
  { label: 'English', value: 'english' },
];
export const OPTIONS_SEARCH_ENGINE = [
  {
    value: 'opensearch',
    label: 'Open Search',
    description: 'A brief description of Open Search',
  },
  {
    value: 'kendra',
    label: 'Kendra',
    description: 'A brief description of Kendra',
  },
];
const KENDRA = OPTIONS_SEARCH_ENGINE[1].value;

const SEARCH_METHOD = [
  { label: 'vector', value: 'vector' },
  { label: 'text', value: 'text' },
  { label: 'mix', value: 'mix' },
];

export default function ModalCreateSession({ dismissModal, modalVisible }) {
  const { addSession } = useSessionStore();
  const { lsLanguageModelList } = useLsLanguageModelList();
  const { lsSessionList, lsGetSessionItem } = useLsSessionList();
  const { appConfigs } = useLsAppConfigs();
  const [name, bindName, resetName, setName] = useInput('');
  const [sessionTemplateOpt, setSessionTemplateOpt] = useState();

  const [displayKendraOptions, setDisplayKendraOptions] = useState(false);
  const [searchEngine, bindSearchEngine, resetSearchEngine, setSearchEngine] =
    useInput(OPTIONS_SEARCH_ENGINE[0].value);

  useEffect(() => {
    setDisplayKendraOptions(searchEngine === KENDRA);
  }, [searchEngine]);

  const [llmData, setLLMData] = useState(lsLanguageModelList[0]);
  const [role, bindRole, resetRole, setRole] = useInput();

  const [language, setLanguage] = useState(OPTIONS_LANGUAGE[0].value);
  useLsAppConfigs();

  const [
    taskDefinition,
    bindTaskDefinition,
    resetTaskDefinition,
    setTaskDefinition,
  ] = useInput();
  const [outputFormat, bindOutputFormat, resetOutputFormat, setOutputFormat] =
    useInput();

  const [
    isCheckedGenerateReport,
    bindGenerateReport,
    resetGenerateReport,
    setIsCheckedGenerateReport,
  ] = useToggle(false, (checked) => {
    if (checked) {
      setIsCheckedContext(false);
      setIsCheckedKnowledgeBase(true);
    }
  });
  const [isCheckedContext, bindContext, resetContext, setIsCheckedContext] =
    useToggle(false);
  const [
    isCheckedKnowledgeBase,
    bindKnowledgeBase,
    resetKnowledgeBase,
    setIsCheckedKnowledgeBase,
  ] = useToggle(true);

  // const [
  // isCheckedMapReduce,
  //   bindMapReduce,
  //   resetMapReduce,
  //   setIsCheckedMapReduce,
  // ] = useToggle(false);

  const [indexName, setIndexName] = useState('');
  const [kendraIndexName, setKendraIndexName] = useState('');
  const { indexNameList, loading: loadingIndexNameList } = useIndexNameList();
  const [searchMethod, setSearchMethod] = useState(SEARCH_METHOD[0].value);
  const [txtDocsNum, setTxtDocsNum] = useState(0);
  const [vecDocsScoreThresholds, setVecDocsScoreThresholds] = useState(0);
  const [txtDocsScoreThresholds, setTxtDocsScoreThresholds] = useState(0);
  const [topK, setTopK] = useState(3);

  const [isCheckedScoreQA, bindScoreQA, resetScoreQA, setIsCheckedScoreQA] =
    useToggle(true);
  const [isCheckedScoreQD, bindScoreQD, resetScoreQD, setIsCheckedScoreQD] =
    useToggle(true);
  const [isCheckedScoreAD, bindScoreAD, resetScoreAD, setIsCheckedScoreAd] =
    useToggle(true);

  const [prompt, setPrompt] = useState('');

  useEffect(() => {
    let partRole = role ? `${role}. ` : '';
    const partTask = taskDefinition ? `${taskDefinition}. ` : '';
    const partOutputFormat = outputFormat || '';
    if (role) {
      if (language === 'english') {
        partRole = `You are a ${partRole}`;
        setPrompt(
          `${partRole}${partTask}${partOutputFormat}\n\nQuestion:{question}\n=========\n{context}\n=========\nAnswer:`
        );
      } else {
        partRole = `你是一名${partRole}`;
        setPrompt(
          `${partRole}${partTask}${partOutputFormat}\n\n问题:{question}\n=========\n{context}\n=========\n答案:`
        );
      }
    }
  }, [role, taskDefinition, outputFormat, language]);

  const sessionData = {
    name,
    searchEngine,
    llmData,
    role,
    language,
    taskDefinition,
    outputFormat,
    isCheckedGenerateReport,
    isCheckedContext,
    isCheckedKnowledgeBase,
    // isCheckedMapReduce,
    indexName,
    kendraIndexName,
    topK,
    searchMethod,
    txtDocsNum,
    vecDocsScoreThresholds,
    txtDocsScoreThresholds,
    isCheckedScoreQA,
    isCheckedScoreQD,
    isCheckedScoreAD,
    prompt,
    tokenContentCheck: appConfigs.tokenContentCheck,
    responseIfNoDocsFound: appConfigs.responseIfNoDocsFound,
  };
  // console.log(sessionData);

  const resetAllFields = useCallback(() => {
    resetName();
    resetSearchEngine();
    setLLMData();
    resetRole();
    resetTaskDefinition();
    resetOutputFormat();
    resetGenerateReport();
    resetContext();
    resetKnowledgeBase();
    // resetMapReduce();
    setIndexName('');
    setKendraIndexName('');
    setSearchMethod(SEARCH_METHOD[0].value);
    setTopK(3);
    setTxtDocsNum(0);
    setVecDocsScoreThresholds(0);
    setTxtDocsScoreThresholds(0);
    // resetIndexName();
    resetScoreQA();
    resetScoreQD();
    resetScoreAD();
    setSessionTemplateOpt(undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (sessionTemplateOpt?.value) {
      const {
        configs: {
          name,
          searchEngine,
          llmData,
          role,
          language,
          taskDefinition,
          outputFormat,
          isCheckedGenerateReport,
          isCheckedContext,
          isCheckedKnowledgeBase,
          // isCheckedMapReduce,
          indexName,
          kendraIndexName,
          topK,
          searchMethod,
          txtDocsNum,
          vecDocsScoreThresholds,
          txtDocsScoreThresholds,
          isCheckedScoreQA,
          isCheckedScoreQD,
          isCheckedScoreAD,
          prompt,
        },
      } = lsGetSessionItem(sessionTemplateOpt.value, lsSessionList);

      // setName(name);
      setSearchEngine(searchEngine);
      setLLMData(llmData);
      setRole(role);
      setLanguage(language);
      setTaskDefinition(taskDefinition);
      setOutputFormat(outputFormat);

      setIsCheckedGenerateReport(isCheckedGenerateReport);
      setIsCheckedContext(isCheckedContext);
      setIsCheckedKnowledgeBase(isCheckedKnowledgeBase);
      // setIsCheckedMapReduce(isCheckedMapReduce);

      setIndexName(indexName);
      setKendraIndexName(kendraIndexName);
      setTopK(topK);
      setSearchMethod(searchMethod);
      setTxtDocsNum(txtDocsNum);
      setVecDocsScoreThresholds(vecDocsScoreThresholds);
      setTxtDocsScoreThresholds(txtDocsScoreThresholds);

      setIsCheckedScoreQA(isCheckedScoreQA);
      setIsCheckedScoreQD(isCheckedScoreQD);
      setIsCheckedScoreAd(isCheckedScoreAD);
    } else {
      setSessionTemplateOpt(undefined);
    }
  }, [sessionTemplateOpt, lsSessionList]);

  const [loading, setLoading] = useState(false);
  return (
    <Modal
      header="Session Configurations"
      onDismiss={dismissModal}
      visible={modalVisible}
      size="large"
      footer={
        <Box>
          <Box float="left">
            <Button onClick={resetAllFields}>Clear Fields</Button>
          </Box>
          <Box float="right">
            <Button
              variant="primary"
              loading={loading}
              iconName="insert-row"
              onClick={async () => {
                try {
                  setLoading(true);
                  await addSession(sessionData);
                  dismissModal();
                  resetAllFields();
                } catch (error) {
                  console.error(error);
                } finally {
                  setLoading(false);
                }
              }}
            >
              Create a session
            </Button>
          </Box>
        </Box>
      }
    >
      <form onSubmit={(e) => e.preventDefault()}>
        <Form>
          <SpaceBetween size={SIZE}>
            <ColumnLayout columns={2}>
              <FormField
                label="Session Name"
                description="Define a session name for future references"
              >
                <Input {...bindName} placeholder="Data search" />
              </FormField>

              {lsSessionList.length === 0 ? null : (
                <FormField
                  label="Refer to an Existing Session"
                  description="Select an existing session as template"
                >
                  <Select
                    selectedOption={sessionTemplateOpt}
                    onChange={({ detail }) =>
                      setSessionTemplateOpt(detail.selectedOption)
                    }
                    options={lsSessionList.map(({ text, sessionId }) => ({
                      value: sessionId,
                      label: text,
                    }))}
                  />
                </FormField>
              )}
            </ColumnLayout>

            <Divider />

            <FormField
              stretch
              label="Engine"
              description="Please select a search engine"
            >
              <Tiles {...bindSearchEngine} items={OPTIONS_SEARCH_ENGINE} />
            </FormField>

            <SpaceBetween direction="vertical" size="xl">
              <ColumnLayout columns={3}>
                <FormField label="LLM" description="Please select an LLM">
                  <Select
                    empty="Add llm if no options present"
                    selectedOption={{ label: llmData?.modelName }}
                    onChange={({ detail }) => {
                      setLLMData(detail.selectedOption.value);
                    }}
                    options={lsLanguageModelList.map((item) => ({
                      label: item.modelName,
                      value: item,
                    }))}
                  />
                </FormField>
                <FormField
                  label="Role Name"
                  description="Please determine the role"
                >
                  <Input {...bindRole} placeholder="a footwear vendor" />
                </FormField>
                <FormField label="Language" description="Select a language">
                  <Select
                    selectedOption={{ value: language }}
                    onChange={({ detail }) =>
                      setLanguage(detail.selectedOption.value)
                    }
                    options={OPTIONS_LANGUAGE}
                  />
                </FormField>
              </ColumnLayout>
              <FormField
                stretch
                label="Task Definition"
                description="Please provide your task definition"
              >
                <Textarea
                  {...bindTaskDefinition}
                  rows={5}
                  placeholder="recommend appropriate footwear to the customer"
                />
              </FormField>
              <FormField
                stretch
                label="Output Format"
                description="Please provide your output format"
              >
                <Textarea
                  {...bindOutputFormat}
                  rows={1}
                  placeholder="answer in English"
                />
              </FormField>

              <SpaceBetween direction="horizontal" size="xxl">
                <FormField constraintText="This is a constraint text">
                  <Toggle {...bindGenerateReport}>Generate Report</Toggle>
                </FormField>
                <FormField constraintText="OFF when generating report">
                  <Toggle disabled={isCheckedGenerateReport} {...bindContext}>
                    Context
                  </Toggle>
                </FormField>
                {/* <FormField constraintText="Can be enabled when generating report">
                  <Toggle
                    disabled={!isCheckedGenerateReport}
                    {...bindMapReduce}
                  >
                    Map Reduce
                  </Toggle>
                </FormField> */}
                <FormField constraintText="ON when generating report">
                  <Toggle
                    {...bindKnowledgeBase}
                    disabled={isCheckedGenerateReport}
                  >
                    Knowledge Base
                  </Toggle>
                </FormField>
              </SpaceBetween>

              {isCheckedKnowledgeBase ? (
                displayKendraOptions ? (
                  <SpaceBetween direction="vertical" size="s">
                    <FormField label="Kendra Index Name" stretch>
                      <Input
                        placeholder="Please provide Kendra index name"
                        value={kendraIndexName}
                        onChange={({ detail }) => {
                          setKendraIndexName(detail.value);
                        }}
                      />
                    </FormField>
                  </SpaceBetween>
                ) : (
                  <SpaceBetween direction="vertical" size="s">
                    <ColumnLayout columns={3}>
                      <FormField label="Index Name" stretch>
                        <Select
                          empty="Upload a file if no options present"
                          onChange={({ detail }) =>
                            setIndexName(detail.selectedOption.value)
                          }
                          loadingText="loading index names"
                          statusType={
                            loadingIndexNameList ? 'loading' : 'finished'
                          }
                          selectedOption={{ value: indexName }}
                          options={indexNameList.map((name) => ({
                            value: name,
                          }))}
                        />
                      </FormField>
                      <FormField stretch label="Search method">
                        <Select
                          selectedOption={{ value: searchMethod }}
                          onChange={({ detail }) =>
                            setSearchMethod(detail.selectedOption.value)
                          }
                          options={SEARCH_METHOD}
                        />
                      </FormField>
                      <FormField
                        stretch
                        label="Threshold for vector search"
                        constraintText="Float number between 0 and 1"
                        errorText={
                          vecDocsScoreThresholds >= 0 &&
                          vecDocsScoreThresholds <= 1
                            ? ''
                            : 'A number between 0 and 1'
                        }
                      >
                        <Input
                          onBlur={() =>
                            vecDocsScoreThresholds !== 0 &&
                            !vecDocsScoreThresholds
                              ? setVecDocsScoreThresholds(0)
                              : null
                          }
                          step={0.01}
                          type="number"
                          value={vecDocsScoreThresholds}
                          onChange={({ detail }) => {
                            setVecDocsScoreThresholds(detail.value);
                          }}
                        />
                      </FormField>
                    </ColumnLayout>
                    <ColumnLayout columns={3}>
                      <FormField
                        stretch
                        label="Number of doc for vector search"
                        constraintText="Integer between 1 and 10"
                        errorText={
                          topK >= 0 && topK <= 10
                            ? ''
                            : 'A number between 0 and 10'
                        }
                      >
                        <Input
                          onBlur={() =>
                            topK !== 0 && !topK ? setTopK(0) : null
                          }
                          step={1}
                          type="number"
                          inputMode="decimal"
                          value={topK}
                          onChange={({ detail }) => {
                            setTopK(detail.value);
                          }}
                        />
                      </FormField>
                      <FormField
                        stretch
                        label="Number of doc for text search"
                        constraintText="Integer between 1 and 10"
                        errorText={
                          txtDocsNum >= 0 && txtDocsNum <= 10
                            ? ''
                            : 'A number between 0 and 10'
                        }
                      >
                        <Input
                          onBlur={() =>
                            txtDocsNum !== 0 && !txtDocsNum
                              ? setTxtDocsNum(0)
                              : null
                          }
                          step={1}
                          type="number"
                          inputMode="decimal"
                          value={txtDocsNum}
                          onChange={({ detail }) => {
                            setTxtDocsNum(detail.value);
                          }}
                        />
                      </FormField>
                      <FormField
                        stretch
                        label="Threshold for text search"
                        constraintText="Float number between 0 and 1"
                        errorText={
                          txtDocsScoreThresholds >= 0 &&
                          txtDocsScoreThresholds <= 1
                            ? ''
                            : 'A number between 0 and 1'
                        }
                      >
                        <Input
                          onBlur={() =>
                            txtDocsScoreThresholds !== 0 &&
                            !txtDocsScoreThresholds
                              ? setTxtDocsScoreThresholds(0)
                              : null
                          }
                          step={0.01}
                          type="number"
                          value={txtDocsScoreThresholds}
                          onChange={({ detail }) => {
                            setTxtDocsScoreThresholds(detail.value);
                          }}
                        />
                      </FormField>
                    </ColumnLayout>
                  </SpaceBetween>
                )
              ) : null}

              {displayKendraOptions ? null : (
                <FormField stretch label="Display Scores">
                  <SpaceBetween direction="horizontal" size="xxl">
                    <Checkbox {...bindScoreQA}>Query-Answer score</Checkbox>
                    <Checkbox {...bindScoreQD}>Query-Doc scores</Checkbox>
                    <Checkbox {...bindScoreAD}>Answer-Doc scores</Checkbox>
                  </SpaceBetween>
                </FormField>
              )}

              <ExpandableSection
                headerText="Prompt Summary"
                defaultExpanded
                headerDescription="Generated automatically by the values of Role Name, Task Definition and Output Format"
              >
                <Textarea value={prompt} disabled rows={6} />
              </ExpandableSection>
            </SpaceBetween>
          </SpaceBetween>
        </Form>
      </form>
    </Modal>
  );
}
