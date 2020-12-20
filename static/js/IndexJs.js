function getInput(){return document.getElementById("input").innerHTML;}
/*
BOW = 0
NGRAM = 2
STYLE = 3
ALL = 1
*/

function POST(endpoint, requestBody){
	const xhr = new XMLHttpRequest();   // new HttpRequest instance 
	xhr.open("POST", endpoint);
	xhr.setRequestHeader("Content-Type", "application/json");
    xhr.send(JSON.stringify(requestBody));

    xhr.onreadystatechange = function () {
      if (this.readyState === 4   && 
		  this.status     ==  200 &&
  	      this.status < 300) {
			  if(this.responseText=="")
				  return
			  result = JSON.parse(this.responseText);
			  console.log(result);
		  }
    }
}

var enums = {
	"BASIC": "BASIC",
	"TF-IDF": "TF-IDF",
	
	"SVC": "SVC",
	"Random Forest": "RF",
	"Multinomial Naive Bayes": "MNB",

	"Character Based": "CHR",
	"Word Based" : "WRD",
	"POS Based" : "POS"

};

var  BaseClasses = ["ALL IN ONE","NGRAM","BOW"]
// Additional: BASIC/TF-IDF SVC/RF/MNB CHR/WRD/POS
function predict(modelNum,enumArray){
	let input = getInput();
	let request = {
		"text": input,
		"modelEnum": modelNum,
		"args": enumArray
	}
	console.log("Sended JSON: ",request);
	POST("/predict",request)

}

function getArgFromElement(element){
	return element.innerText.split("\n")[0];
}
function getTag(element){
	return element.getElementsByTagName("a")[0];
}
function getParent(element){
	return element.parentNode;
}

function click(event){
	currentElement = event.target; 
	current =  getArgFromElement(currentElement);
	args = [current];

	while(current!="ALL IN ONE" && currentElement!=null){
		if(currentElement.tagName=="MENUITEM" && !args.includes(current))
			args.push(current);

		currentElement = getParent(currentElement);
		current = getArgFromElement(currentElement);
	}

	

	let Modelname = args.pop();
	let modelNum = BaseClasses.findIndex((element) => element == Modelname)
	let enumArray = []
	if(args.length!=0){
		for(var i = 0;i<args.length;i++){
			enumArray.push(enums[args[i]])
		}
	}
	predict(modelNum,enumArray);



}
var items = document.getElementsByTagName("a");

for(var i=0;i<items.length;i++){
	items[i].onclick = click;
}



